# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image

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


def download_tinyimagenet(root):
    if os.path.isdir(root) and len(os.listdir(root)) > 0:
        print('Download not needed, files already on disk.')
    else:
        from onedrivedownloader import download

        print('Downloading dataset')
        ln = "https://unimore365-my.sharepoint.com/:u:/g/personal/263133_unimore_it/EVKugslStrtNpyLGbgrhjaABqRHcE3PB_r2OEaV7Jy94oQ?e=9K29aD"
        download(ln, filename=os.path.join(root, 'tiny-imagenet-processed.zip'), unzip=True, unzip_path=root, clean=True)


def load_data(root, train=True):
    data = []
    for num in range(20):
        data.append(np.load(os.path.join(
            root, 'processed/x_%s_%02d.npy' %
                  ('train' if train else 'val', num + 1))))
    data = np.concatenate(np.array(data))

    targets = []
    for num in range(20):
        targets.append(np.load(os.path.join(
            root, 'processed/y_%s_%02d.npy' %
                  ('train' if train else 'val', num + 1))))
    targets = np.concatenate(np.array(targets))

    return data, targets


class TrainTinyImagenet(MammothDataset):
    def __init__(self, root: str, transform, not_aug_transform, drift_transform) -> None:
        super().__init__()
        self.transform = transform
        self.not_aug_transform = not_aug_transform
        self.drift_transform = drift_transform

        download_tinyimagenet(root)
        self.data, self.targets = load_data(root, train=True)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))

        if target in self.drifted_classes:
            img = self.drift_transform(img)

        original_img = img.copy()
        img = self.transform(img)
        not_aug_img = self.not_aug_transform(original_img)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img

    def __len__(self):
        return len(self.data)

    def select_classes(self, classes_list: list[int]):
        if len(classes_list) == 0:
            self.data = np.array([])
            self.targets = np.array([])
            self.classes = []
            return

        mask = np.zeros_like(self.targets)
        for label in classes_list:
            mask = np.logical_or(mask, self.targets == label)
        self.data = self.data[mask]
        self.targets = self.targets[mask]

        self.classes = classes_list

    def apply_drift(self, classes: list):
        if len(set(self.classes).union(classes)) == 0:
            return
        self.drifted_classes.extend(classes)

    def prepare_normal_data(self):
        pass


class TestTinyImagenet(MammothDataset):
    def __init__(self, root: str, transform, drift_transform) -> None:
        super().__init__()

        self.transform = transform
        self.drift_transform = drift_transform

        download_tinyimagenet(root)
        self.data, self.targets = load_data(root, train=False)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))

        if target in self.drifted_classes:
            img = self.drift_transform(img)

        img = self.transform(img)

        if hasattr(self, 'logits'):
            return img, target, self.logits[index]

        return img, target

    def __len__(self):
        return len(self.data)

    def select_classes(self, classes_list: list[int]):
        if len(classes_list) == 0:
            self.data = np.array([])
            self.targets = np.array([])
            self.classes = []
            return

        mask = np.zeros_like(self.targets)
        for label in classes_list:
            mask = np.logical_or(mask, self.targets == label)
        self.data = self.data[mask]
        self.targets = self.targets[mask]

        self.classes = classes_list

    def apply_drift(self, classes: list):
        if len(set(self.classes).union(classes)) == 0:
            return
        self.drifted_classes.extend(classes)

    def prepare_normal_data(self):
        pass


class SequentialTinyImagenet(ContinualDataset):

    NAME = 'seq-tinyimg'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
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
        DRIFT_SEVERITY = self.args.drift_severity

        DRIFT = transforms.Compose([
            self.DRIFT_TYPES[self.args.concept_drift](DRIFT_SEVERITY),
            transforms.ToPILImage()
        ])

        if train:
            return TrainTinyImagenet(base_path() + 'TINYIMG',
                                     transform=self.TRANSFORM, not_aug_transform=self.TRANSFORM, drift_transform=DRIFT)
        else:
            return TestTinyImagenet(base_path() + 'TINYIMG',
                                    transform=self.TRANSFORM, drift_transform=DRIFT)

    @staticmethod
    def get_backbone():
        return resnet18(SequentialTinyImagenet.N_CLASSES_PER_TASK
                        * SequentialTinyImagenet.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4802, 0.4480, 0.3975),
                                (0.2770, 0.2691, 0.2821))
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialTinyImagenet.get_batch_size()
