# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--n_epochs', type=int,
                        help='Batch size.')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size.')

    parser.add_argument('--distributed', type=str, default='no', choices=['no', 'dp', 'ddp'])

    parser.add_argument('--concept_drift', default=-1, choices=[-1, 0, 1, 2, 3, 4, 5, 6], type=int,
                        help='Choose the drift transform to be applied to drifting data: \
                        Defocus Blur-> 0, Gaussian Noise-> 1, Shot Noise-> 2, Speckle Noise-> 3, \
                        Rotation -> 4, Pixel Permutation -> 5, Identity (No transform) -> 6 or -1')
    parser.add_argument('--drift_severity', default=1, choices=[1, 2, 3, 4, 5], type=int,
                        help='Choose the intensity of the drift transform:')
    parser.add_argument('--drift_adaptation', default=0, choices=[0, 1, 2], type=int,
                        help='Choose adaptation method when concept drift is detected: \
                        0 -> No adaptation, 1 -> Full relearning, 2 -> Buffer resampling')
    # parser.add_argument('--n_slots', default=None, type=int, 
    #                     help='number of classes per task used when generating task stream randomly based on slots')
    parser.add_argument('--n_drifts', default=None, type=int, 
                        help='number of drifts created when creating evenly spaced drfits')
    parser.add_argument('--max_classes_per_drift', type=int, default=0,
                        help='maximum number of classes that can be drifted at once. Used only with n_drifts. \
                        If set to 0 (default), all previous classes will drift.')
    parser.add_argument('--sequential_drifts', action='store_true', 
                        help='if used each task will consist of both new classes and \
                        drifted classes from previous task')


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--non_verbose', default=0, choices=[0, 1], type=int, help='Make progress bars non verbose')
    parser.add_argument('--disable_log', default=0, choices=[0, 1], type=int, help='Enable csv logging')

    parser.add_argument('--validation', default=0, choices=[0, 1], type=int,
                        help='Test on the validation set')
    parser.add_argument('--ignore_other_metrics', default=0, choices=[0, 1], type=int,
                        help='disable additional metrics')
    parser.add_argument('--debug_mode', type=int, default=0, help='Run only a few forward steps per epoch')
    parser.add_argument('--nowand', default=0, choices=[0, 1], type=int, help='Inhibit wandb logging')
    parser.add_argument('--wandb_entity', type=str, default='regaz', help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, default='mammoth', help='Wandb project name')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int,
                        help='The batch size of the memory buffer.')
    parser.add_argument('--buffer_mode', default='balanced', type=str,
                        choices=['ring', 'reservoir', 'balanced'], 
                        help='The method for buffer sampling.')
