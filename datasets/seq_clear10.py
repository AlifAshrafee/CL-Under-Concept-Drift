from typing import Tuple, List
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from backbone.ResNet18 import resnet18
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import ContinualDataset
from datasets.mammoth_dataset import MammothDataset
from utils.conf import base_path_dataset as base_path


CLASS_NAMES = [
    "baseball", "bus", "camera", "cosplay", "dress",
    "hockey", "laptop", "racing", "soccer", "sweater",
]

# Maps classes to appropriate temporal bucket pairs
# classes in each task are consisted of 2 consecutive buckets with ~2 years of data
TEMPORAL_BUCKETS = {
    'regular': [1, 2],  # regular task buckets 1-2 (~2004-2005)
    'drift': [9, 10],  # drifted task buckets 9-10 (~2012-2014)
}


def download_clear10(root):
    train_dir = os.path.join(root, "train_image_only", "labeled_images", "1")
    test_dir = os.path.join(root, "test", "labeled_images", "1")

    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        if len(os.listdir(train_dir)) > 0 and len(os.listdir(test_dir)) > 0:
            print("CLEAR-10 dataset already downloaded!")
            return

    CLEAR10_TRAIN_URL = "https://huggingface.co/datasets/elvishelvis6/CLEAR-Continual_Learning_Benchmark/resolve/main/clear10-train-image-only.zip"
    CLEAR10_TEST_URL = "https://huggingface.co/datasets/elvishelvis6/CLEAR-Continual_Learning_Benchmark/resolve/main/clear10-test.zip"
    print("CLEAR-10 dataset not found!")
    print("Please manually download CLEAR-10 from the following links:")
    print(f"Training Set: {CLEAR10_TRAIN_URL}")
    print(f"Test Set: {CLEAR10_TEST_URL}")
    print(f"And then extract to: {root}/")
    return


def load_images_from_buckets(base_path: str, buckets: List[int], classes: List[int]):
    data, targets = [], []

    for class_id in classes:
        class_name = CLASS_NAMES[class_id]
        for bucket in buckets:
            bucket_dir = os.path.join(base_path, str(bucket), class_name)
            if not os.path.exists(bucket_dir):
                print(f"Warning: Directory not found: {bucket_dir}")
                continue

            for img_name in os.listdir(bucket_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(bucket_dir, img_name)
                    img = Image.open(img_path).convert("RGB")
                    data.append(np.array(img))
                    targets.append(class_id)

    return data, np.array(targets) if targets else ([], np.array([]))


class CLEAR10(MammothDataset, Dataset):
    """
    Base class for CLEAR-10 datasets.
    Handles temporal bucket-based data loading for both train and test splits.
    """
    def __init__(self, root: str, train: bool = True):
        download_clear10(root)
        self.root = root
        self.train = train
        self.buckets = TEMPORAL_BUCKETS['regular']  # default to regular buckets from earlier years
        self.classes = []
        self.drifted_classes = []

        self.data = []
        self.targets = np.array([])

        split = "train_image_only" if train else "test"
        self.base_images_path = os.path.join(root, split, "labeled_images")

    def __len__(self):
        return len(self.targets)

    def select_classes(self, classes_list: List[int]):
        if not classes_list:
            self.data = []
            self.targets = np.array([])
            self.classes = []
            return

        self.classes = classes_list
        self.data, self.targets = load_images_from_buckets(self.base_images_path, 
                                                           self.buckets, classes_list)

    def apply_drift(self, recurring_classes: List[int]):
        """
        Applies temporal drift by replacing recurring 
        drifted class data with samples from later buckets
        """
        drift_buckets = TEMPORAL_BUCKETS['drift']  # using latest buckets (9-10) for drift severity
        dataloader_classes = np.unique(np.array(self.targets))
        drifting_classes = [cls for cls in dataloader_classes if cls in recurring_classes]
        if not drifting_classes:
            print("No classes to drift in current test loader")
            return

        for cls in drifting_classes:
            if cls not in self.drifted_classes:
                self.drifted_classes.append(cls)

        keep_indices = [i for i, t in enumerate(self.targets) if t not in drifting_classes]
        kept_data = [self.data[i] for i in keep_indices]
        kept_targets = self.targets[keep_indices] if keep_indices else np.array([])

        new_data, new_targets = load_images_from_buckets(self.base_images_path,
                                                         drift_buckets, drifting_classes)

        if len(new_data) > 0:
            if len(kept_data) > 0:
                self.data = kept_data + new_data
                self.targets = np.concatenate((kept_targets, new_targets))
            else:
                self.data = new_data
                self.targets = new_targets
        elif len(kept_data) > 0:
            self.data = kept_data
            self.targets = kept_targets

    def prepare_normal_data(self):
        pass


class TrainCLEAR10(CLEAR10):
    def __init__(self, root: str, transform, not_aug_transform):
        super().__init__(root, train=True)
        self.transform = transform
        self.not_aug_transform = not_aug_transform

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img, mode="RGB")

        original_img = img.copy()
        img = self.transform(img)
        not_aug_img = self.not_aug_transform(original_img)

        if hasattr(self, "logits"):
            return img, target, not_aug_img, self.logits[index]
        return img, target, not_aug_img


class TestCLEAR10(CLEAR10):
    def __init__(self, root: str, transform) -> None:
        super().__init__(root, train=False)
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img, mode="RGB")
        img = self.transform(img)
        return img, target


class BufferTransform:
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return x

        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        return transform(x)


class SequentialCLEAR10(ContinualDataset):
    NAME = "seq-clear10"
    SETTING = "class-il"
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5

    TRANSFORM = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]
    )

    def get_dataset(self, train=True):
        if train:
            return TrainCLEAR10(base_path() + "CLEAR", transform=self.TRANSFORM, 
                                not_aug_transform=self.TRANSFORM)
        return TestCLEAR10(base_path() + "CLEAR", transform=self.TRANSFORM)

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCLEAR10.N_CLASSES_PER_TASK 
                        * SequentialCLEAR10.N_TASKS)

    @staticmethod
    def get_transform():
        return BufferTransform()

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize([0.485, 0.456, 0.406], 
                                    [0.229, 0.224, 0.225])

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    @staticmethod
    def get_loss():
        return torch.nn.CrossEntropyLoss()

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
        return 32
