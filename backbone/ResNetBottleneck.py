import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(conv1x1(in_planes, self.expansion * planes, stride), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(conv1x1(in_planes, planes * self.expansion, stride), nn.BatchNorm2d(planes * self.expansion))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks: List[int], num_classes: int, nf: int = 64):
        super(ResNet, self).__init__()
        self.in_planes = nf

        self.conv1 = conv3x3(3, nf)
        self.bn1 = nn.BatchNorm2d(nf)
        self.layer1 = self._make_layer(block, nf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.fc = nn.Linear(nf * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Functions to instantiate different ResNet versions
def resnet18(nclasses, nf=64):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)

def resnet34(nclasses, nf=64):
    return ResNet(BasicBlock, [3, 4, 6, 3], nclasses, nf)

def resnet50(nclasses, nf=64):
    return ResNet(Bottleneck, [3, 4, 6, 3], nclasses, nf)

def resnet101(nclasses, nf=64):
    return ResNet(Bottleneck, [3, 4, 23, 3], nclasses, nf)

def resnet152(nclasses, nf=64):
    return ResNet(Bottleneck, [3, 8, 36, 3], nclasses, nf)

def test_resnet(model_name: str, input_shape: Tuple[int, int, int], num_classes: int) -> None:

    batch_size = 32  # Define batch size for testing
    inputs = torch.randn(batch_size, *input_shape)  # Create random input tensor

    model_map = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152
    }

    if model_name not in model_map:
        raise ValueError(f"Unknown model name: {model_name}. Please choose from {list(model_map.keys())}.")

    model = model_map[model_name](nclasses=num_classes, nf=64)
    model.eval()
    outputs = model(inputs)

    # Check if the output has the expected shape (batch_size, num_classes)
    assert outputs.shape == (batch_size, num_classes), f"Output shape mismatch: {outputs.shape}"

    print(f"{model_name} test passed!")

if __name__ == "__main__":
    # Example tests for different ResNet models with CIFAR-10 (3 channels, 32x32 input) and 10 output classes
    test_resnet(model_name="resnet18", input_shape=(3, 32, 32), num_classes=10)
    test_resnet(model_name="resnet34", input_shape=(3, 32, 32), num_classes=10)
    test_resnet(model_name="resnet50", input_shape=(3, 32, 32), num_classes=10)
    test_resnet(model_name="resnet101", input_shape=(3, 32, 32), num_classes=10)
    test_resnet(model_name="resnet152", input_shape=(3, 32, 32), num_classes=10)
