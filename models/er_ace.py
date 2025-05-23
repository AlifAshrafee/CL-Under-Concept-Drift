# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from datasets import get_dataset

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class ErACE(ContinualModel):
    NAME = 'er_ace'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(ErACE, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device, mode=args.buffer_mode)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = get_dataset(args).N_TASKS * get_dataset(args).N_CLASSES_PER_TASK
        self.task = 0

    def end_task(self, dataset):
        self.task += 1

    def observe(self, inputs, labels, not_aug_inputs):

        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits = self.net(inputs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        self.opt.zero_grad()
        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1

        if self.task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        loss = self.loss(logits, labels)
        loss_re = torch.tensor(0.)

        if self.task > 0 and not self.buffer.is_empty():
            buf_data = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buf_inputs, buf_labels = buf_data[0], buf_data[1]
            loss_re = self.loss(self.net(buf_inputs), buf_labels)

        loss += loss_re

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        return loss.item()

    def buffer_resampling(self, dataset, flagged_classes):
        for cls in flagged_classes:
            flushed_indices = self.buffer.flush_class(cls)
            num_samples_needed = len(flushed_indices)
            buffer_resampling_data_loader = dataset.request_drifted_data(cls, num_samples_needed)

            data_iter = iter(buffer_resampling_data_loader)
            inputs, labels, not_aug_inputs = next(data_iter)
            assert inputs.shape[0] == num_samples_needed, f"Requested {num_samples_needed} samples but received {labels.shape[0]}"

            for i, index in enumerate(flushed_indices):
                self.buffer.num_seen_examples += 1
                self.buffer.current_size = min(self.buffer.current_size + 1, self.buffer.buffer_size)
                self.buffer.examples[index] = not_aug_inputs[i].to(self.buffer.device)
                self.buffer.labels[index] = labels[i].to(self.buffer.device)

            print(f"Class {cls} samples replaced: {num_samples_needed}")
