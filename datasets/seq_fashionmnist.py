
from typing import Tuple
from PIL import Image

import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from torchvision.datasets import FashionMNIST
import numpy as np
import torch.optim
import torch.nn.functional as F

from utils.conf import base_path_dataset as base_path
from datasets.utils.continual_dataset import ContinualDataset
from datasets.transforms.to_thre_channels import ToThreeChannels
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


class TrainFashionMNIST(MammothDataset, FashionMNIST):
    def __init__(self, root, transform, not_aug_transform, drift_transform, download=False):
        super().__init__(root, train=True, transform=transform, target_transform=None, download=download)
        self.not_aug_transform = not_aug_transform
        self.drift_transform = drift_transform
        self.classes = list(range(10))

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

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


class TestFashionMNIST(MammothDataset, FashionMNIST):
    def __init__(self, root, transform, drift_transform, download=False):
        super().__init__(root, train=False, transform=transform, target_transform=None, download=download)
        self.drift_transform = drift_transform
        self.classes = list(range(10))

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if target in self.drifted_classes:
            img = self.drift_transform(img)

        img = self.transform(img)

        if hasattr(self, 'logits'):
            return img, target, self.logits[index]

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


class SequentialFashionMNIST(ContinualDataset):

    NAME = 'seq-fashionmnist'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5

    TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        ToThreeChannels(),
    ])

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
        DRIFT_SEVERITY = self.args.drift_severity

        DRIFT = transforms.Compose([
            self.DRIFT_TYPES[self.args.concept_drift](DRIFT_SEVERITY),
            transforms.ToPILImage()
        ])

        if train:
            return TrainFashionMNIST(base_path() + 'FASHIONMNIST',
                                    transform=self.TRANSFORM, not_aug_transform=self.TRANSFORM, drift_transform=DRIFT,
                                    download=True)
        else:
            return TestFashionMNIST(base_path() + 'FASHIONMNIST',
                                    transform=self.TRANSFORM, drift_transform=DRIFT,
                                    download=True)

    def get_transform(self):
        return transforms.Compose([transforms.ToPILImage(), self.TRANSFORM])

    @staticmethod
    def get_backbone():
        return resnet18(SequentialFashionMNIST.N_CLASSES_PER_TASK
                        * SequentialFashionMNIST.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialFashionMNIST.get_batch_size()

    @staticmethod
    def get_scheduler(model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
        return scheduler
