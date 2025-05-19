# Copyright 2020-present, Tao Zhuo
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from copy import deepcopy

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Strong Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    return parser


def tf_tensor(xs, transforms):
    device = xs.device
    xs = torch.cat([transforms(x).unsqueeze_(0) for x in xs.cpu()], dim=0)
    return xs.to(device=device)


class SER(ContinualModel):
    NAME = 'ser'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone, loss, args, transform):
        super(SER, self).__init__(backbone, loss, args, transform)
        self.net_old = None
        self.buffer = Buffer(self.args.buffer_size, self.device, mode=args.buffer_mode)

    def end_task(self, dataset):
        self.net_old = deepcopy(self.net)
        self.net_old.eval()

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()

        inputs_aug = tf_tensor(inputs, self.transform)
        outputs = self.net(inputs_aug)
        loss = F.cross_entropy(outputs, labels)

        if self.net_old is not None:
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(inputs.size(0), transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += F.cross_entropy(buf_outputs, buf_labels)

            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            outputs_old = self.net_old(inputs_aug)
            loss += self.args.beta * F.mse_loss(outputs, outputs_old)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=inputs, labels=labels, logits=outputs.data)

        return loss.item()

    def buffer_resampling(self, dataset, flagged_classes):
        for cls in flagged_classes:
            flushed_indices = self.buffer.flush_class(cls)
            num_samples_needed = len(flushed_indices)
            buffer_resampling_data_loader = dataset.request_drifted_data(cls, num_samples_needed)

            data_iter = iter(buffer_resampling_data_loader)
            inputs, labels, not_aug_inputs = next(data_iter)
            outputs = self.net(inputs.to(self.device)).data
            assert inputs.shape[0] == num_samples_needed, f"Requested {num_samples_needed} samples but received {labels.shape[0]}"

            for i, index in enumerate(flushed_indices):
                self.buffer.num_seen_examples += 1
                self.buffer.current_size = min(self.buffer.current_size + 1, self.buffer.buffer_size)
                self.buffer.examples[index] = not_aug_inputs[i].to(self.buffer.device)
                self.buffer.labels[index] = labels[i].to(self.buffer.device)
                self.buffer.logits[index] = outputs[i].to(self.buffer.device)

            print(f"Class {cls} samples replaced: {num_samples_needed}")
