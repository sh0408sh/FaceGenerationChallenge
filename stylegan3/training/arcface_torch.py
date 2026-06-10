"""Differentiable ArcFace-style face embedding network.

The InsightFace ONNX runtime path is useful for evaluation, but it does not
participate in PyTorch autograd.  This module provides the common InsightFace
IResNet-50 backbone so frozen ArcFace embeddings can be used as a training loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    def __init__(self, layers, dropout=0, num_features=512):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * 7 * 7, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, eps=1e-05),
            )
        layers = [IBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(IBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return F.normalize(x, dim=1)


def iresnet50():
    return IResNet([3, 4, 14, 3])


def _strip_prefix(state_dict, prefix):
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key[len(prefix):] if key.startswith(prefix) else key: value for key, value in state_dict.items()}


def load_iresnet50(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ['state_dict', 'model_state_dict', 'backbone', 'net']:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'Unsupported ArcFace checkpoint format: {path}')
    state_dict = _strip_prefix(checkpoint, 'module.')
    state_dict = _strip_prefix(state_dict, 'backbone.')
    state_dict = _strip_prefix(state_dict, 'model.')
    model = iresnet50().eval().requires_grad_(False).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 20 or len(unexpected) > 20:
        raise RuntimeError(
            f'ArcFace checkpoint does not look compatible with IResNet-50: '
            f'{len(missing)} missing, {len(unexpected)} unexpected keys')
    return model


class ArcFaceEmbedder(nn.Module):
    def __init__(self, checkpoint_path, device):
        super().__init__()
        self.model = load_iresnet50(checkpoint_path, device)

    def forward(self, img):
        x = img.to(torch.float32)
        # Generated faces are already roughly aligned/cropped.  Keep this path
        # differentiable by using only tensor resize and channel normalization.
        if x.shape[2] != 112 or x.shape[3] != 112:
            x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        x = x.clamp(-1, 1)
        return self.model(x)
