# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image
from torchvision.datasets import CIFAR100
import numpy as np

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import ContinualDataset
from utils.conf import base_path_dataset as base_path
from datasets.transforms.driftTransforms import (
    DefocusBlur,
    GaussianNoise,
    ShotNoise,
    SpeckleNoise,
    RotateTransform,
    PixelPermutation,
    Identity,
)
from datasets.mammoth_dataset import MammothDataset


class TrainCIFAR100(MammothDataset, CIFAR100):
    def __init__(self, root: str, transform, not_aug_transform, drift_transform) -> None:
        self.root = root    # Workaround to avoid printing the already downloaded messages
        super().__init__(root, train=True, transform=transform, target_transform=None, download=not self._check_integrity())
        self.not_aug_transform = not_aug_transform
        self.drift_transform = drift_transform
        self.classes = list(range(100))

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')

        if target in self.drifted_classes:
            img = self.drift_transform(img)

        original_img = img.copy()
        img = self.transform(img)
        not_aug_img = self.not_aug_transform(original_img)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img

    def select_classes(self, classes_list: list[int]):
        if len(classes_list) == 0:
            self.data = np.array([])
            self.targets = np.array([])
            self.classes = []
            return

        mask = np.zeros_like(np.array(self.targets))
        for label in classes_list:
            mask = np.logical_or(mask, np.array(self.targets) == label)
        self.data = self.data[mask]
        self.targets = np.array(self.targets)[mask]

        self.classes = classes_list

    def apply_drift(self, classes: list):
        if len(set(self.classes).union(classes)) == 0:
            return
        self.drifted_classes.extend(classes)

    def prepare_normal_data(self):
        pass


class TestCIFAR100(MammothDataset, CIFAR100):
    """Workaround to avoid printing the already downloaded messages."""

    def __init__(self, root, transform, drift_transform) -> None:
        self.root = root
        super().__init__(root, train=False, transform=transform, target_transform=None, download=not self._check_integrity())
        self.drift_transform = drift_transform
        self.classes = list(range(100))

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if target in self.drifted_classes:
            img = self.drift_transform(img)

        img = self.transform(img)

        return img, target

    def select_classes(self, classes_list: list[int]):
        if len(classes_list) == 0:
            self.data = np.array([])
            self.targets = np.array([])
            self.classes = []
            return

        mask = np.zeros_like(np.array(self.targets))
        for label in classes_list:
            mask = np.logical_or(mask, np.array(self.targets) == label)
        self.data = self.data[mask]
        self.targets = np.array(self.targets)[mask]

        self.classes = classes_list

    def apply_drift(self, classes: list):
        if len(set(self.classes).union(classes)) == 0:
            return
        self.drifted_classes.extend(classes)

    def prepare_normal_data(self):
        pass


class SequentialCIFAR100(ContinualDataset):

    NAME = 'seq-cifar100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10

    TRANSFORM = transforms.Compose([transforms.ToTensor()])

    DRIFT_TYPES = [
        DefocusBlur,
        GaussianNoise,
        ShotNoise,
        SpeckleNoise,
        RotateTransform,
        PixelPermutation,
        Identity,
    ]

    def get_dataset(self, train=True):
        """returns native version of represented dataset"""
        DRIFT_SEVERITY = self.args.drift_severity

        DRIFT = transforms.Compose([
            self.DRIFT_TYPES[self.args.concept_drift](DRIFT_SEVERITY),
            transforms.ToPILImage()
        ])

        if train:
            return TrainCIFAR100(base_path() + 'CIFAR100',
                                 transform=self.TRANSFORM, not_aug_transform=self.TRANSFORM, drift_transform=DRIFT)
        else:
            return TestCIFAR100(base_path() + 'CIFAR100',
                                transform=self.TRANSFORM, drift_transform=DRIFT)

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR100.N_CLASSES_PER_TASK
                        * SequentialCIFAR100.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_scheduler(model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.SGD(
            model.net.parameters(),
            lr=args.lr,
            weight_decay=args.optim_wd,
            momentum=args.optim_mom,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
        return scheduler

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCIFAR100.get_batch_size()
