import torch
from torch import nn
from torchvision import models
from .common import CLASSES

NAMES = ('resnet18', 'vgg16', 'mobilenet_v2')
SPEC = 'cifar10-native32-v1'


class CIFARModel(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.register_buffer('mean', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1))
        if architecture == 'resnet18':
            net = models.resnet18(weights=None, num_classes=10)
            net.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        elif architecture == 'vgg16':
            net = models.vgg16(weights=None, num_classes=10)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            net.classifier = nn.Linear(512, 10)
        elif architecture == 'mobilenet_v2':
            net = models.mobilenet_v2(weights=None, num_classes=10)
            net.features[0][0].stride = (1, 1)
        else:
            raise ValueError(f'Unknown architecture: {architecture}')
        self.network = net

    def forward(self, x):
        return self.network((x - self.mean) / self.std)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('spec') != SPEC or checkpoint.get('classes') != CLASSES:
        raise ValueError('Expected a transferlab CIFAR-10 checkpoint with matching class order')
    model = CIFARModel(checkpoint['architecture'])
    for key in ('mean', 'std'):
        if not torch.equal(checkpoint['state_dict'].get(key, torch.empty(0)), getattr(model, key)):
            raise ValueError('Checkpoint normalization does not match the CIFAR-10 specification')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(device).eval(), checkpoint
