from collections import defaultdict


class FLOPsCounter:
    def __init__(self):
        self.flops_per_task = defaultdict(int)
        self.current_task = 0
        self.total_flops = 0

    def update(self, flops):
        self.flops_per_task[self.current_task] += flops
        self.total_flops += flops

    def report(self):
        print("\nFLOPs Summary:")
        for task, flops in self.flops_per_task.items():
            print(f"Task {task}: {flops/1e9:.2f} GFLOPs")
        print(f"Total: {self.total_flops/1e9:.2f} GFLOPs")
