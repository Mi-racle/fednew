from collections import Counter
from typing import List, Optional, Union

import numpy as np
import torch

from datasets import Dataset, DatasetDict
from flwr_datasets.partitioner import Partitioner
from torch.utils.data import Subset


class DirichletPartitioner(Partitioner):
    """Partition according to the Dirichlet distribution.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    alpha: float
        Parameter of the Dirichlet distribution
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """

    def __init__(
            self,
            num_clients: int,
            alpha: float = 0.5,
            seed: int = 42,
    ):
        super().__init__()
        self.num_clients = num_clients
        self.alpha = alpha
        self.seed = seed
        self.split_datasets: Optional[List[Dataset]] = None

    def partition(self):

        min_required_samples_per_client = 10
        min_samples = 0
        prng = np.random.default_rng(self.seed)

        # get the targets
        labels = np.array(self.dataset['label'])
        num_classes = len(Counter(labels))
        total_samples = self.dataset.num_rows
        idx_clients: List[List] = []
        while min_samples < min_required_samples_per_client:
            idx_clients = [[] for _ in range(self.num_clients)]
            for k in range(num_classes):
                idx_k = np.where(labels == k)[0]
                prng.shuffle(idx_k)
                proportions = prng.dirichlet(np.repeat(self.alpha, self.num_clients))
                proportions = np.array(
                    [
                        p * (len(idx_j) < total_samples / self.num_clients)
                        for p, idx_j in zip(proportions, idx_clients)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_k_split = np.split(idx_k, proportions)
                idx_clients = [
                    idx_j + idx.tolist() for idx_j, idx in zip(idx_clients, idx_k_split)
                ]
                min_samples = min([len(idx_j) for idx_j in idx_clients])

        # trainsets_per_client = [Subset(self.dataset, idxs) for idxs in idx_clients]
        trainsets_per_client = [self.dataset.select(idxs) for idxs in idx_clients]

        output_details(trainsets_per_client, num_classes, 'dirichlet')

        return trainsets_per_client

    def load_partition(self, node_id: int) -> Dataset:
        """Load a single partition based on the partition index.

        Parameters
        ----------
        node_id : int
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition : Dataset
            single dataset partition
        """
        if self.split_datasets is None:
            self.split_datasets = self.partition()
        return self.split_datasets[node_id]


class LabelPartitioner(Partitioner):
    """Partition the data according to the number of labels per client.

    Logic from https://github.com/Xtra-Computing/NIID-Bench/.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    num_labels_per_client: int
        Number of labels per client
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42
    dataset_name : str
        Name of the dataset to be used

    Returns
    -------
    Tuple[List[Subset], Dataset]
"""

    def __init__(
            self,
            num_clients: int,
            labels_per_client: int = 3,
            seed: int = 42,
    ):
        super().__init__()
        self.num_clients = num_clients
        self.labels_per_client = labels_per_client
        self.seed = seed
        self.split_datasets: Optional[List[Dataset]] = None

    def partition(self):

        prng = np.random.default_rng(self.seed)

        labels = np.array(self.dataset['label'])
        num_classes = len(Counter(labels))
        times = [0 for _ in range(num_classes)]
        contains = []

        for i in range(self.num_clients):
            # current = [i % num_classes]
            # times[i % num_classes] += 1
            # j = 1
            current = []
            j = 0
            while j < self.labels_per_client:
                pool = [index for index, item in enumerate(times) if item == 0]
                pool = num_classes if len(pool) == 0 else pool
                index = prng.choice(pool, 1)[0]
                if index not in current:
                    current.append(index)
                    times[index] += 1
                    j += 1
            contains.append(current)
        idx_clients: List[List] = [[] for _ in range(self.num_clients)]
        for i in range(num_classes):
            idx_k = np.where(labels == i)[0]
            prng.shuffle(idx_k)
            idx_k_split = np.array_split(idx_k, times[i])
            ids = 0
            for j in range(self.num_clients):
                if i in contains[j]:
                    idx_clients[j] += idx_k_split[ids].tolist()
                    ids += 1
        trainsets_per_client = [self.dataset.select(idxs) for idxs in idx_clients]

        output_details(trainsets_per_client, num_classes, 'label')

        return trainsets_per_client

    def load_partition(self, node_id: int) -> Dataset:
        """Load a single partition based on the partition index.

        Parameters
        ----------
        node_id : int
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition : Dataset
            single dataset partition
        """
        if self.split_datasets is None:
            self.split_datasets = self.partition()
        return self.split_datasets[node_id]


def output_details(datasets: list[Union[dict, Dataset, DatasetDict]], num_classes: int, name: str):

    # row: dataset; col: class
    count_table = []

    for dataset in datasets:
        class_counts = []
        labels = np.array(dataset['label'])
        for i in range(num_classes):
            idx_k = np.where(labels == i)[0]
            count = len(idx_k)
            class_counts.append(count)
        count_table.append(class_counts)

    count_table = np.array(count_table)
    np.savetxt(f'dataset_stats_{name}.csv', count_table, delimiter=',')
