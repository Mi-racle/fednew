import json
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()

INPUT_CHANNELS = {
    'mnist': 1,
    'medmnistS': 1,
    'medmnistC': 1,
    'medmnistA': 1,
    'covid19': 3,
    'fmnist': 1,
    'emnist': 1,
    'femnist': 1,
    'cifar10': 3,
    'cinic10': 3,
    'svhn': 3,
    'cifar100': 3,
    'celeba': 3,
    'usps': 1,
    'tiny_imagenet': 3,
    'domain': 3,
}


def _get_domain_classes_num():
    try:
        with open(PROJECT_DIR / 'data' / 'domain' / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        return metadata['class_num']
    except:
        return 0


def _get_synthetic_classes_num():
    try:
        with open(PROJECT_DIR / 'data' / 'synthetic' / 'args.json', 'r') as f:
            metadata = json.load(f)
        return metadata['class_num']
    except:
        return 0


NUM_CLASSES = {
    'mnist': 10,
    'medmnistS': 11,
    'medmnistC': 11,
    'medmnistA': 11,
    'fmnist': 10,
    'svhn': 10,
    'emnist': 62,
    'femnist': 62,
    'cifar10': 10,
    'cinic10': 10,
    'cifar100': 100,
    'covid19': 4,
    'usps': 10,
    'celeba': 2,
    'tiny_imagenet': 200,
    'synthetic': _get_synthetic_classes_num(),
    'domain': _get_domain_classes_num(),
}


class DecoupledModel(nn.Module):
    def __init__(self):
        super(DecoupledModel, self).__init__()
        self.need_all_features_flag = False
        self.all_features = []
        self.base: nn.Module = None
        self.classifier: nn.Module = None
        self.dropout: List[nn.Module] = []

    def need_all_features(self):
        target_modules = [
            module
            for module in self.base.modules()
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
        ]

        def get_feature_hook_fn(model, input, output):
            if self.need_all_features_flag:
                self.all_features.append(output.clone().detach())

        for module in target_modules:
            module.register_forward_hook(get_feature_hook_fn)

    def check_avaliability(self):
        if self.base is None or self.classifier is None:
            raise RuntimeError(
                "You need to re-write the base and classifier in your custom model class."
            )
        self.dropout = [
            module
            for module in list(self.base.modules()) + list(self.classifier.modules())
            if isinstance(module, nn.Dropout)
        ]

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.base(x))

    def get_final_features(self, x: Tensor, detach=True) -> Tensor:
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        out = self.base(x)

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return func(out)

    def get_all_features(self, x: Tensor) -> Optional[List[Tensor]]:
        feature_list = None
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        self.need_all_features_flag = True
        _ = self.base(x)
        self.need_all_features_flag = False

        if len(self.all_features) > 0:
            feature_list = self.all_features
            self.all_features = []

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return feature_list


class LeNet5(DecoupledModel):
    def __init__(self, dataset: str) -> None:
        super(LeNet5, self).__init__()
        feature_length = {
            "mnist": 256,
            "medmnistS": 256,
            "medmnistC": 256,
            "medmnistA": 256,
            "covid19": 49184,
            "fmnist": 256,
            "emnist": 256,
            "femnist": 256,
            "cifar10": 400,
            "cinic10": 400,
            "svhn": 400,
            "cifar100": 400,
            "celeba": 33456,
            "usps": 200,
            "tiny_imagenet": 2704,
        }
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(INPUT_CHANNELS[dataset], 6, 5),
                bn1=nn.BatchNorm2d(6),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(6, 16, 5),
                bn2=nn.BatchNorm2d(16),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(feature_length[dataset], 120),
                activation3=nn.ReLU(),
                fc2=nn.Linear(120, 84),
            )
        )

        self.classifier = nn.Linear(84, NUM_CLASSES[dataset])

    def forward(self, x):
        return self.classifier(F.relu(self.base(x)))
