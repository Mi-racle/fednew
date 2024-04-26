from collections import OrderedDict
from copy import deepcopy
from logging import INFO
from typing import Dict, Tuple, List, Union

import flwr as fl
import torch
import torch.nn as nn
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from flwr.common import Metrics
from flwr.common.logger import log
from flwr.common.typing import Scalar
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

    def get_parameters(self, config):
        return [val.cpu().numpy() for name, val in self.model.state_dict().items() if 'bn' not in name]

    def fit(self, parameters, config):
        """
        config == fit_config() + proximal_mu
        """
        self.cluster_models = set_params(self.model, parameters, self.cid)

        # cifar batch 64
        valloader = DataLoader(self.valset, batch_size=64, drop_last=True)

        distribution = self.evaluate_distribution(valloader)

        trainloader = DataLoader(self.trainset, batch_size=config['batch'], shuffle=True, drop_last=True)

        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), {'distribution': distribution}

    def evaluate(self, parameters, config):
        self.cluster_models = set_params(self.model, parameters, self.cid)

        if self.cid == 0:
            torch.save(self.model.state_dict(), 'best.pt')

        # Read from config
        batch, epochs, patience, server_round, proximal_mu = \
            config['batch_size'], config['epochs'], config['patience'], config['server_round'], config['proximal_mu']

        # Construct dataloader
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True, drop_last=True)

        # Define optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        # Train
        self.train(trainloader, optimizer, epochs=epochs, patience=patience, device=self.device)

        loss, accuracy = self.test(valloader, device=self.device)

        # cifar batch 64
        valloader = DataLoader(self.valset, batch_size=64, drop_last=True)

        # Evaluate
        loss, accuracy = self.test(valloader, device=self.device)

        # Return statistics
        return float(loss), len(valloader.dataset), {'loss': float(loss), 'accuracy': float(accuracy)}

    def train(self, trainloader, optim, epochs, patience, device: Union[str, torch.device]):
        """Train the network on the training set."""
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(epochs):
            for batch in trainloader:
                image_key = 'image' if 'image' in batch else 'img'
                images, labels = batch[image_key].to(device), batch['label'].to(device)
                optim.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
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
