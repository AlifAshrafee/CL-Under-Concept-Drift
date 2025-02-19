from collections import defaultdict


class TimeCounter:
    def __init__(self):
        self.time_per_task = defaultdict(int)
        self.current_task = 0
        self.total_time = 0

    def update(self, time):
        self.time_per_task[self.current_task] += time
        self.total_time += time

    def report(self):
        print("\nTime Summary:")
        for task, time in self.time_per_task.items():
            print(f"Task {task}: {time:.2f} (s)")
        print(f"Total: {self.total_time:.2f} (s)")
