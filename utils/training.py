# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import sklearn.metrics
import sys
from argparse import Namespace
from typing import Tuple
import json
from datetime import datetime
import torch
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from drift_detection import detect_uncertainty_drift
from utils.loggers import *
from utils.status import ProgressBar
from utils.feature_shift_logging import extract_features
import time
from utils.flops_counter import FLOPsCounter
from utils.time_counter import TimeCounter
from torch.profiler import profile, record_function, ProfilerActivity

try:
    import wandb
except ImportError:
    wandb = None


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
            dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')

def get_unique_labels(data_loader):
    """
    Method for checking the unique labels in a data loader
    :param data_loader: the data loader
    :return: the unique labels
    """
    labels = []
    for data in data_loader:
        batch_labels = data[1]
        labels.extend(batch_labels.numpy())

    unique_labels = torch.unique(torch.tensor(labels))
    return unique_labels.tolist()

def count_flops_time(model, train_loader):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
    ) as prof:
        with record_function("model_inference"):
            inputs, labels, not_aug_inputs = next(iter(train_loader))
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            not_aug_inputs = not_aug_inputs.to(model.device)
            start_time = time.time()
            loss = model.meta_observe(inputs, labels, not_aug_inputs)
            total_time = time.time() - start_time
            assert not math.isnan(loss)

    total_flops = sum(event.flops for event in prof.key_averages() if hasattr(event, 'flops') and event.flops > 0)
    return total_flops * len(train_loader), total_time * len(train_loader)

def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes, f1_scores = [], [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        predictions = list()
        all_labels = list()

        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data[0], data[1]
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)

                _, pred = torch.max(outputs.data, 1)

                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                predictions.append(pred)
                all_labels.append(labels)

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
        predictions = torch.cat(predictions, dim=0).cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

        f1 = sklearn.metrics.f1_score(all_labels, predictions, average='macro') * 100.0
        f1_scores.append(f1)

    model.net.train(status)
    return accs, accs_mask_classes, f1_scores


def train(model: ContinualModel, dataset: ContinualDataset, args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    print(args)

    if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        args.wandb_url = wandb.run.get_url()

    model.net.to(model.device)
    results, results_mask_classes, results_f1 = [], [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    if not args.ignore_other_metrics:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            if args.concept_drift == -1:
                _, _ = dataset_copy.get_data_loaders()
            else:
                _, _ = dataset_copy.get_drifted_data_loaders()
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            random_results_class, random_results_task = evaluate(model, dataset_copy)

    print(file=sys.stderr)
    flops_counter = FLOPsCounter()
    time_counter = TimeCounter()
    for t in range(dataset.N_TASKS):
        flops_counter.current_task = t
        time_counter.current_task = t
        model.net.train()
        if args.concept_drift == -1:
            train_loader, _ = dataset.get_data_loaders()
        else:
            train_loader, _ = dataset.get_drifted_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t and not args.ignore_other_metrics:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]
        if args.drift_adaptation > 0:
            flagged_classes = detect_uncertainty_drift(dataset, model)
            if len(flagged_classes) > 0:
                if args.drift_adaptation == 1:
                    train_loader = dataset.request_drifted_data_with_current_data(flagged_classes)
                elif args.drift_adaptation == 2:
                    model.buffer_resampling(dataset, flagged_classes)

        scheduler = dataset.get_scheduler(model, args)
        for epoch in range(model.args.n_epochs):
            if args.model == 'joint':
                continue
            if epoch == 0:
                total_flops, total_time = count_flops_time(model, train_loader)
                flops_counter.update(total_flops)
                time_counter.update(total_time)

            for i, data in enumerate(train_loader):
                if args.debug_mode and i > 3:
                    break

                inputs, labels, not_aug_inputs = data[0], data[1], data[2]
                inputs = inputs.to(model.device)
                labels = labels.to(model.device)
                not_aug_inputs = not_aug_inputs.to(model.device)

                if hasattr(dataset.train_loader.dataset, 'logits'):
                    logits = data[-1]
                    logits = logits.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, logits)
                else:
                    loss = model.meta_observe(inputs, labels, not_aug_inputs)
                assert not math.isnan(loss)
                progress_bar.prog(i, len(train_loader), epoch, t, loss)

            if scheduler is not None:
                scheduler.step()

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        flops_counter.report()
        time_counter.report()
        metrics = evaluate(model, dataset)
        accs = metrics[:2]
        results.append(accs[0])
        results_mask_classes.append(accs[1])
        results_f1.append(metrics[2])

        mean_acc = np.mean(accs, axis=1)
        mean_f1_scores = np.mean(metrics[2])
        print_mean_accuracy(mean_acc, mean_f1_scores, t + 1, dataset.SETTING)

        if args.feature_logging:
            extract_features(model, dataset.test_loaders[0], f"features-{str(args.dataset)}-cd-{str(args.concept_drift)}-n-{str(args.n_drifts)}-T-{t + 1}.pth")

        if not args.disable_log:
            logger.log(mean_acc)
            logger.log_fullacc(accs)

        if not args.nowand:
            d2 = {'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1],
                  **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
                  **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])}}

            wandb.log(d2)

    log_filename = (
        f"../results/Concept-Drift/{datetime.now().strftime('%m-%d-%y-%H-%M-%S')}-{args.dataset}-{args.model}-buf-{args.buffer_size}"
        f"{'-drift-' + str(args.concept_drift) + '-s-' + str(args.drift_severity) + '-n-' + str(args.n_drifts) + '-adaptation-' + str(args.drift_adaptation) if args.concept_drift > -1 else '-no-drift'}.json"
    )

    with open(log_filename, 'w') as jsonfile:
        json.dump({'task_accuracies': results, 'task_f1': results_f1}, jsonfile)

    if not args.disable_log and not args.ignore_other_metrics:
        logger.add_bwt(results, results_mask_classes)
        logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            logger.add_fwt(results, random_results_class,
                           results_mask_classes, random_results_task)

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()
