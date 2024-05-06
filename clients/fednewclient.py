import json
from collections import OrderedDict
from copy import deepcopy
from functools import reduce
from logging import INFO
from typing import Dict, Tuple, List, Union

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from flwr.common import Metrics, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.common.typing import Scalar, NDArrays
from flwr_datasets import FederatedDataset
from net import resnet18
from torch.utils.data import DataLoader
from utils import test, apply_transforms

NETWORK = resnet18(pretrained=False, in_channels=3, num_classes=53)
NUM_LAYERS = 27


# Flower client, adapted from Pytorch quickstart example
class FedNewClient(fl.client.NumPyClient):

    cluster_models = []

    def __init__(self, trainset, valset, cid):

        self.trainset = trainset
        self.valset = valset
        self.cid = cid

        # Instantiate model
        self.model = NETWORK

        # Determine device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)  # send model to device

        self.last_distribution = None
        self.current_distribution = None

        self.batch_size = 32
        self.epochs = 5
        self.patience = 10
        self.server_round = 100
        self.proximal_mu = 0.1

    def get_parameters(self, config):
        return [val.cpu().numpy() for name, val in self.model.state_dict().items() if 'bn' not in name]

    def fit(self, parameters, config):
        """
        config == fit_config() + proximal_mu
        """
        self.batch_size, self.epochs, self.patience, self.server_round, self.proximal_mu = \
            config['batch_size'], config['epochs'], config['patience'], config['server_round'], config['proximal_mu']

        self.cluster_models = set_params(self.model, parameters, self.cid)

        # cifar batch 64
        valloader = DataLoader(self.valset, batch_size=64, drop_last=True)

        self.last_distribution = self.current_distribution
        self.current_distribution = self.evaluate_distribution(valloader)

        trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), {'distribution': self.current_distribution}

    def evaluate(self, parameters, config):
        self.cluster_models = set_params(self.model, parameters, self.cid)

        if self.cid == 0:
            torch.save(self.model.state_dict(), 'best.pt')

        # Construct dataloader
        trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # Define optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        # Train
        self.train(
            trainloader,
            optimizer,
            epochs=self.epochs,
            patience=self.patience,
            proximal_mu=self.proximal_mu,
            device=self.device
        )

        # cifar batch 64
        valloader = DataLoader(self.valset, batch_size=64, drop_last=True)

        # Evaluate
        loss, accuracy = self.test(valloader, device=self.device)

        # Return statistics
        return float(loss), len(valloader.dataset), {'loss': float(loss), 'accuracy': float(accuracy)}

    def train(self, trainloader, optim, epochs, patience, proximal_mu, device: Union[str, torch.device]):
        """Train the network on the training set."""
        criterion = nn.CrossEntropyLoss()
        euclidean = nn.MSELoss()
        weighted_weights = []
        for cluster_id, model in enumerate(self.cluster_models):
            weights = [val.cpu().numpy() for name, val in model.state_dict().items() if 'bn' not in name]
            weights = np.array(weights)
            weighted_weights.append(
                [
                    layer * self.current_distribution[cluster_id] for layer in weights
                ]
            )
        global_model: NDArrays = [
            reduce(np.add, layer_updates) / sum(self.current_distribution)
            for layer_updates in zip(*weighted_weights)
        ]
        self.model.train()
        for _ in range(epochs):
            for batch in trainloader:
                image_key = 'image' if 'image' in batch else 'img'
                images, labels = batch[image_key].to(device), batch['label'].to(device)
                optim.zero_grad()
                loss = criterion(self.model(images), labels)
                if self.last_distribution is not None:
                    loss += euclidean(self.last_distribution, self.current_distribution)
                loss.backward()
                for w, w_t in zip(self.model, global_model):
                    w.grad.data += proximal_mu * (w.data - w_t.data)
                optim.step()

    def test(self, testloader, device: Union[str, torch.device]):
        """Validate the network on the entire test set."""
        criterion = nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        self.model.eval()
        with torch.no_grad():
            for data in testloader:
                image_key = 'image' if 'image' in data else 'img'
                images, labels = data[image_key].to(device), data['label'].to(device)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        return loss, accuracy

    def evaluate_distribution(self, testloader):

        criterion = nn.CrossEntropyLoss()
        clustered_loss = []

        for cluster_model in self.cluster_models:

            cluster_model.eval()
            batched_loss = []

            with torch.no_grad():

                for data in testloader:

                    image_key = 'image' if 'image' in data else 'img'
                    images, labels = data[image_key].to(self.device), data['label'].to(self.device)
                    outputs = cluster_model(images)
                    loss = criterion(outputs, labels).item()
                    batched_loss.append(loss)

            clustered_loss.append(batched_loss)

        clustered_loss = torch.tensor(clustered_loss)
        _, indices = torch.min(clustered_loss, dim=0)

        distribution = []

        for cluster_id in range(len(self.cluster_models)):

            count = (indices == cluster_id).sum().item()
            distribution.append(count)

        return distribution


def get_new_client_fn(dataset: FederatedDataset):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        # Let's get the partition corresponding to the i-th client
        client_dataset = dataset.load_partition(int(cid), 'train')

        client_dataset_splits = client_dataset.train_test_split(test_size=0.2)

        trainset = client_dataset_splits['train']
        valset = client_dataset_splits['test']

        # Now we apply the transform to each batch.
        trainset = trainset.with_transform(apply_transforms)
        valset = valset.with_transform(apply_transforms)

        # Create and return client
        return FedNewClient(trainset, valset, int(cid)).to_client()

    return client_fn


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        'epochs': 5,  # Number of local epochs done by clients
        'batch_size': 32,  # Batch size to use by clients during fit()
        'patience': 5,  # early stopping (not working currently)
        'server_round': server_round
    }
    return config


def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays], cid: int):
    """Set model weights from a list of NumPy ndarrays."""
    keys = [k for k in model.state_dict().keys() if 'bn' not in k]

    if len(params) % NUM_LAYERS != 0:
        raise 'Cluster params returned by server are in wrong format'

    cluster_num = len(params) // NUM_LAYERS
    models = []

    for cluster_id in range(cluster_num):

        cluster_params = params[cluster_id * NUM_LAYERS: (cluster_id + 1) * NUM_LAYERS]
        params_dict = zip(keys, cluster_params)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        copy_model = deepcopy(model)
        copy_model.load_state_dict(state_dict, strict=False)
        models.append(copy_model)

    return models


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m['accuracy'] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    acc = sum(accuracies) / sum(examples)
    log(INFO, f"Decentralized acc: {acc}")
    # Aggregate and return custom metric (weighted average)
    return {'accuracy': acc}


def get_new_evaluate_fn(
        centralized_testset: Dataset,
):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ):
        """Use the entire CIFAR-10 test set for evaluation."""

        # Determine device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = NETWORK
        set_params(model, parameters, 0)
        model.to(device)

        # Apply transform to dataset
        testset = centralized_testset.with_transform(apply_transforms)

        # Disable tqdm for dataset preprocessing
        disable_progress_bar()

        testloader = DataLoader(testset, batch_size=50)
        loss, accuracy, cm = test(model, testloader, device=device)

        return loss, {'accuracy': accuracy}

    return evaluate
