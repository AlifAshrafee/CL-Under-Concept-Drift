# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from typing import Tuple

import numpy as np
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from datasets.stream_spec import StreamSpecification
from datasets.mammoth_dataset import MammothDataset


class ContinualDataset:
    """
    Continual learning evaluation setting.
    """
    NAME: str
    SETTING: str
    N_CLASSES_PER_TASK: int
    N_TASKS: int

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args
        self.seen_classes = []
        self.recurring_classes = []

        if not all((self.NAME, self.SETTING, self.N_CLASSES_PER_TASK, self.N_TASKS)):
            raise NotImplementedError(
                'The dataset must be initialized with all the required fields.')

        if args.n_drifts or args.sequential_drifts:
            n_tasks = self.N_TASKS
            n_classes = self.N_CLASSES_PER_TASK * self.N_TASKS
            self.stream_spec = StreamSpecification(
                n_tasks,
                n_classes,
                random_seed=args.seed,
                # n_slots=args.n_slots,     # For future implementations
                n_drifts=args.n_drifts,
                sequential_drifts=args.sequential_drifts,
                max_classes_per_drift=args.max_classes_per_drift,
            )
            self.stream_spec_it = iter(self.stream_spec)

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        new_classes = list(range(self.i, self.i + self.N_CLASSES_PER_TASK))
        train_dataset = self.get_dataset(train=True)
        train_dataset.select_classes(new_classes)
        test_dataset = self.get_dataset(train=False)
        test_dataset.select_classes(new_classes)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        self.train_loader = train_loader
        self.test_loaders.append(test_loader)

        self.i += self.N_CLASSES_PER_TASK
        return train_loader, test_loader

    def get_drifted_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        current_classes = next(self.stream_spec_it)

        recurring_classes = list(set(self.seen_classes) & set(current_classes))
        self.recurring_classes = recurring_classes

        new_classes = list(set(current_classes) - set(self.seen_classes))
        self.seen_classes.extend(new_classes)

        print(f"New Classes: {new_classes}")
        print(f"Recurring Classes: {recurring_classes}")

        train_dataset = self.get_dataset(train=True)
        train_dataset.select_classes(new_classes)
        train_dataset.prepare_normal_data()
        test_dataset = self.get_dataset(train=False)
        test_dataset.select_classes(new_classes)
        test_dataset.prepare_normal_data()

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        self.train_loader = train_loader

        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        self.test_loaders.append(test_loader)

        if len(recurring_classes) > 0:
            for t in range(len(self.test_loaders)):
                prev_test_data = self.test_loaders[t].dataset
                prev_test_data.apply_drift(recurring_classes)
                self.test_loaders[t] = DataLoader(prev_test_data, batch_size=self.args.batch_size, 
                                                  shuffle=False, num_workers=4)

        return train_loader, test_loader

    def request_drifted_data_with_current_data(self, drifted_classes) -> DataLoader:
        drifting_train_dataset = self.get_dataset(train=True)
        drifting_train_dataset.select_classes(drifted_classes)
        drifting_train_dataset.apply_drift(drifted_classes)
        train_dataset = torch.utils.data.ConcatDataset([drifting_train_dataset, self.train_loader.dataset])
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        self.train_loader = train_loader
        return train_loader

    def request_drifted_data(self, drifted_classes) -> DataLoader:
        drifting_train_dataset = self.get_dataset(train=True)
        drifting_train_dataset.select_classes(drifted_classes)
        drifting_train_dataset.apply_drift(drifted_classes)
        buffer_resampling_data_loader = DataLoader(drifting_train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        return buffer_resampling_data_loader

    def get_dataset(self, train=True) -> MammothDataset:
        """returns native version of represented dataset"""
        raise NotImplementedError

    @staticmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_transform() -> nn.Module:
        """
        Returns the transform to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_loss() -> nn.Module:
        """
        Returns the loss to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_normalization_transform() -> nn.Module:
        """
        Returns the transform used for normalizing the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_denormalization_transform() -> nn.Module:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_scheduler(model, args: Namespace) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Returns the scheduler to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_epochs():
        raise NotImplementedError

    @staticmethod
    def get_batch_size():
        raise NotImplementedError

    @staticmethod
    def get_minibatch_size():
        raise NotImplementedError
