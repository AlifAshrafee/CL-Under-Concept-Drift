import numpy as np
import collections
from typing import Union


class StreamSpecification:

    def __init__(
        self,
        n_tasks: int,
        n_classes: int,
        random_seed: int = None,
        # n_slots: Union[int, None] = None,     # For future implementations
        n_drifts: Union[int, None] = None,
        sequential_drifts: bool = False,
        max_classes_per_drift: int = 0,
    ) -> None:
        """
        Utility class that represents the class-task layout of the overall task stream.

        Args:
        n_tasks - number of tasks in the stream
        n_classes - number of all classes in dataset
        random_seed - random seed used for experiments
        n_slots - number of classes per task, used for creating stream layout in the style of Class-Incremental learning with repetition
        n_drifts - number of drifts created in whole task stream
        sequential_drifts - if true each task will consist of set of new classes and drifted classes from previous task
        max_classes_per_drifts - number of classes that drift is applied to. used only with n_drifts
        """
        assert n_tasks >= 2, 'need at least two tasks for continual learning'
        assert n_drifts is not None or sequential_drifts, 'must specify drift type'

        # if n_slots is not None:
        #     assert n_drifts == None and sequential_drifts == False, 'you must use n_slots, n_drifts or sequential_drifts arguments (cant use them together)'
        if n_drifts is not None:
            assert sequential_drifts == False, 'you must use n_drifts or sequential_drifts arguments (cant use them together)'
        if sequential_drifts:
            assert n_drifts is None, 'you must use n_drifts or sequential_drifts arguments (cant use them together)'

        self.n_tasks = n_tasks
        # self.n_slots = n_slots
        self.n_classes = n_classes
        self.random_seed = random_seed
        self.n_drifts = n_drifts
        self.sequential_drifts = sequential_drifts
        self.max_classes_per_drift = max_classes_per_drift

        self._new_classes: list[list[int]] = list()
        # Current implementation of task stream does not require having drifted classes separately
        # self._drifted_classes: list[list[int]] = list()
        if n_drifts is not None:
            self.create_n_drifts()
        elif sequential_drifts:
            self.create_sequential_drifts()
        # else:
        #     self.random_class_asigment()

        self.current_task = 0

    def create_n_drifts(self):
        assert self.n_drifts < self.n_tasks

        drift_duration = self.n_tasks // (self.n_drifts + 1)
        drift_indexes = list(np.linspace(drift_duration, self.n_tasks, num=self.n_drifts, endpoint=False, dtype=int))
        print('Creating drifts at tasks: ', drift_indexes)

        classes_per_task = self.n_classes // self.n_tasks
        drift_begin_idx = 0
        for t in range(self.n_tasks):
            new_classes = list(range(classes_per_task * t, classes_per_task * (t+1)))
            if t in drift_indexes:
                drifted_classes = list(range(drift_begin_idx, classes_per_task * t))
                drifted_classes = drifted_classes[-min(self.max_classes_per_drift, len(drifted_classes)):]
                self._new_classes.append(drifted_classes + new_classes)
                drift_begin_idx = classes_per_task * t
            else:
                self._new_classes.append(new_classes)

    def create_sequential_drifts(self):
        classes_per_task = self.n_classes // self.n_tasks
        assert classes_per_task >= 2, 'At least two classes should be present in each new task'

        for t in range(self.n_tasks):
            new_classes = list(range(classes_per_task * t, classes_per_task * (t + 1)))
            drifted_classes = list(range(max(classes_per_task * t - classes_per_task, 0), classes_per_task * t))
            self._new_classes.append(drifted_classes + new_classes)

    # def random_class_asigment(self):
    #     """ Random class assigment to the tasks.
    #     Based on the notion of slot-based generator from https://arxiv.org/pdf/2301.11396
    #     Each task has n_slots slots. Each slot can be filled with exatly one class.
    #     When classes apears more than once in the stream we assume, that drift occured, and data of given class should change.
    #     When there are some classes left at the end, they are appended to the end of the stream.
    #     """
    #     classes_pool = list(range(self.n_classes))
    #     used_classes = set()
    #     random_state = np.random.RandomState(self.random_seed)

    #     for _ in range(self.n_tasks):
    #         if len(classes_pool) < self.n_slots:
    #             classes_pool.extend(range(self.n_classes))

    #         task_classes = [None]
    #         label = None
    #         for _ in range(self.n_slots):
    #             while label in task_classes:
    #                 label = random_state.choice(classes_pool, replace=False)
    #             task_classes.append(label)
    #             classes_pool.remove(label)
    #             if all(l in set(task_classes) for l in classes_pool):
    #                 break
    #         task_classes.remove(None)

    #         self._new_classes.append([])
    #         self._drifted_classes.append([])
    #         for c in task_classes:
    #             if c not in used_classes:
    #                 self._new_classes[-1].append(c)
    #                 used_classes.add(c)
    #             else:
    #                 self._drifted_classes[-1].append(c)

    #     class_mapping = dict()
    #     last_class = 0
    #     for task_class in self._new_classes:
    #         for c in task_class:
    #             if c not in class_mapping:
    #                 class_mapping[c] = last_class
    #                 last_class += 1

    #     self._new_classes = [sorted(class_mapping[c] for c in task_class) for task_class in self._new_classes]
    #     self._drifted_classes = [sorted(class_mapping[c] for c in task_class) for task_class in self._drifted_classes]

    def __next__(self):
        while self.current_task < self.n_tasks:
            yield self._new_classes[self.current_task]
            self.current_task += 1

    def __iter__(self):
        return next(self)

    @property
    def all_drifted_classes(self):
        """return all classes for which drift has occured so far"""
        all_drifted = set()
        for task_drifted in self._drifted_classes[:self.current_task+1]:
            for c in task_drifted:
                all_drifted.add(c)
        return list(all_drifted)

    @property
    def drifted_classes(self):
        """return classes from last task for which dirft has occured"""
        return self._drifted_classes[self.current_task]

    @property
    def new_classes(self):
        """return classes from current task"""
        return self._new_classes[self.current_task]

    @property
    def max_drifts_per_class(self):
        if len(self._drifted_classes) == 0:
            print('max_drifts_per_class call: no drifts in current setup')
            return 0
        counter = collections.Counter()
        for classes in self._drifted_classes:
            counter.update(classes)
        max_occurence = counter.most_common(1)[0][1]
        return max_occurence


if __name__ == '__main__':
    ss = StreamSpecification(20, 100, 45, n_drifts=10)
    ssi = iter(ss)
    for new, old in ssi:
        print(f'new = {new} old = {old}')
