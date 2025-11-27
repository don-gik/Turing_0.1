import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import logging

from .config import *


class DataManager:
    def __init__(self):
        self.trainTransform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        ])

        self.testTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        ])

        self.log = logging.getLogger(__name__)

        self.init()

    def init(self):
        self.log.info(f"Dataset CIFAR100 is getting prepared in directory {ROOT}")
        

        self.trainSet = datasets.CIFAR100(
            root=ROOT,
            train=True,
            download=True,
            transform=self.trainTransform
        )

        self.testSet = datasets.CIFAR100(
            root=ROOT,
            train=False,
            download=True,
            transform=self.testTransform
        )

        self.log.info("Dataset is prepared.")

    def get_data(self):
        self.log.info(f"Dataset fetched from {ROOT}")

        return (self.trainSet, self.testSet)