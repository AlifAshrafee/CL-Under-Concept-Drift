# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

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


class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Er, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device, mode=args.buffer_mode)

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_data = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buf_inputs, buf_labels = buf_data[0], buf_data[1]
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels[:real_batch_size])

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
