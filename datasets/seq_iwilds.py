from PIL import Image
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset, store_masked_loaders, store_drifted_masked_loaders)
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights

import copy
import collections
import functools


@functools.cache
def get_class_details() -> Tuple[dict[int, list], collections.Counter]:
    dataset = get_dataset(dataset="iwildcam", download=True)
    class_traps_ids = collections.defaultdict(set)
    class_counts = collections.Counter()

    for _, label, metadata in dataset:
        label = label.item()
        trap_id = metadata[0].item()
        class_traps_ids[label].add(trap_id)
        class_counts.update([label])

    for label, traps_ids in class_traps_ids.items():
        class_traps_ids[label] = list(traps_ids)

    return class_traps_ids, class_counts


@functools.cache
def get_valid_classes():
    """return list of classes that have at last 500 samples and at least 2 domains"""
    class_traps_ids, class_counts = get_class_details()
    selected_classes = [label for label in class_traps_ids if len(class_traps_ids[label]) > 2 and class_counts[label] >= 500]
    return selected_classes


class TrainIWilds:
    def __init__(self, transform, not_aug_transform, target_transform) -> None:
        dataset = get_dataset(dataset="iwildcam", download=True)
        self.transforms = transform
        self.not_aug_transform = not_aug_transform
        self.target_transform = target_transform
        train_data = dataset.get_subset("train")
        self.subset = torch.utils.data.Subset(train_data, list(range(len(train_data))))

    def select_domains(self, class_traps: dict[list]) -> None:
        selected_samples = list()
        for i, (_, label, metadata) in enumerate(self.subset):
            if metadata[0].item() in class_traps[label.item()]:
                selected_samples.append(i)
        self.subset.indices = selected_samples

    def select_classes(self, condiction: callable) -> None:
        selected_samples = list()
        for i, (_, label, _) in enumerate(self.subset):
            if label.item() in self.target_transform and condiction(self.target_transform[label.item()]):
                selected_samples.append(i)
        self.subset.indices = selected_samples

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        not_aug_img, label, _ = self.subset[index]

        img = self.transforms(not_aug_img)
        label = self.target_transform[label.item()]
        not_aug_img = self.not_aug_transform(not_aug_img)
        return img, label, not_aug_img

    def __len__(self):
        return len(self.subset)


class TestIWilds(TrainIWilds):
    def __init__(self, transform, target_transform) -> None:
        super().__init__(transform, None, target_transform)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        not_aug_img, label, _ = self.subset[index]

        img = self.transforms(not_aug_img)
        label = self.target_transform[label.item()]
        return img, label


class SequentialIWilds(ContinualDataset):
    NAME = 'seq-iwilds'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 5
    N_TASKS = 14
    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # TODO these are ImageNet norm stats
    ])
    NOT_AUG_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        valid_classes = sorted(get_valid_classes())
        self.class_mapping = {i: j for i, j in zip(valid_classes, range(len(valid_classes)))}

        class_traps_ids, _ = get_class_details()
        self.class_normal_traps = collections.defaultdict(list)
        self.class_drift_traps = collections.defaultdict(list)
        for label, trap_ids in class_traps_ids.items():
            if label not in valid_classes:
                continue
            i = len(trap_ids) // 2
            self.class_normal_traps[label].extend(trap_ids[:i])
            self.class_drift_traps[label].extend(trap_ids[i:])

    def get_drifted_data_loaders(self, args):
        train_dataset = TrainIWilds(self.TRANSFORM, self.NOT_AUG_TRANSFORM, self.class_mapping)
        test_dataset = TestIWilds(self.TEST_TRANSFORM, self.class_mapping)
        drifting_train_dataset = copy.deepcopy(train_dataset)
        drifting_test_dataset = copy.deepcopy(test_dataset)

        train_dataset.select_domains(self.class_normal_traps)
        test_dataset.select_domains(self.class_normal_traps)
        drifting_train_dataset.select_domains(self.class_drift_traps)
        drifting_test_dataset.select_domains(self.class_drift_traps)

        train, test = self.store_drifted_masked_loaders(train_dataset=train_dataset, test_dataset=test_dataset,
                                                        drifting_train_dataset=drifting_train_dataset,
                                                        drifting_test_dataset=drifting_test_dataset)

        return train, test

    def store_drifted_masked_loaders(self: ContinualDataset, train_dataset: Dataset, test_dataset: Dataset, drifting_train_dataset: Dataset,
                                     drifting_test_dataset: Dataset) -> Tuple[DataLoader, DataLoader]:
        def drift_mask(label): return label >= self.i - self.N_CLASSES_PER_TASK and label < self.i
        def normal_mask(label): return label >= self.i and label < self.i + self.N_CLASSES_PER_TASK

        drifting_train_dataset.select_classes(drift_mask)
        drifting_test_dataset.select_classes(drift_mask)
        train_dataset.select_classes(normal_mask)
        test_dataset.select_classes(normal_mask)

        combined_train_dataset = torch.utils.data.ConcatDataset([drifting_train_dataset, train_dataset])
        train_loader = DataLoader(combined_train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        self.train_loader = train_loader

        if self.i > 0:
            drifted_test_loader = DataLoader(drifting_test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)
            assert len(self.test_loaders) > 0
            self.test_loaders[len(self.test_loaders) - 1] = drifted_test_loader   # replacing previous testloader with drifted images

        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)

        # current test loader contains undrifted images
        self.test_loaders.append(test_loader)

        self.i += self.N_CLASSES_PER_TASK
        return train_loader, test_loader

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialIWilds.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        n_classes = SequentialIWilds.N_CLASSES_PER_TASK * SequentialIWilds.N_TASKS
        net.fc = nn.Linear(net.fc.weight.shape[1], n_classes)
        return net

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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
        return SequentialIWilds.get_batch_size()
