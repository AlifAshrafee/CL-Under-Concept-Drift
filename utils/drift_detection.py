import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from functools import partial
from alibi_detect.cd import ClassifierUncertaintyDrift


def initialize_uncertainty_detector(ref_data, device):
    encoding_dim = 32
    clf = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    clf.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    clf.maxpool = nn.Identity()
    num_features = clf.fc.in_features
    clf.fc = nn.Linear(num_features, encoding_dim)
    clf = clf.to(device).eval()
    cd = ClassifierUncertaintyDrift(ref_data, model=clf, backend="pytorch", p_val=0.05, preds_type="logits")

    return cd


def detect_uncertainty_drift(dataset, model) -> list:
    recurring_classes = dataset.recurring_classes
    if len(recurring_classes) == 0:
        return []

    labels = ["No!", "Yes!"]
    flagged_classes = []

    for cls in recurring_classes:
        filtered_images = []
        for test_loader in dataset.test_loaders:
            for batch in test_loader:
                img_batch, target_batch = batch[0], batch[1]
                mask = target_batch == cls
                selected_images = img_batch[mask]
                filtered_images.append(selected_images)

        new_images = torch.cat(filtered_images, dim=0)
        if new_images.size(0) > 0 and hasattr(model, "buffer"):
            ref_samples = model.buffer.get_class_data(cls)
            if not isinstance(ref_samples, int):
                drift_detector = initialize_uncertainty_detector(ref_samples, model.device)
                preds = drift_detector.predict(new_images)
                print(f"Drift in class {cls}? {labels[preds['data']['is_drift']]}")
                print(f"Feature-wise p-values: {', '.join([f'{p_val:.3f}' for p_val in preds['data']['p_val']])}")

                if preds["data"]["is_drift"]:
                    flagged_classes.append(cls)

    return flagged_classes
