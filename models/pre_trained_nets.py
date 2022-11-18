import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def get_pre_trained_model(model_name, pretrained=True, n_classes=1, require_all_grads=False):
    if model_name == 'resnet50':
        return ResNet50(n_classes=n_classes, pretrained=pretrained,
                        require_all_grads=require_all_grads)
    elif model_name == 'mobilenet':
        return MobileNet(n_classes=n_classes, pretrained=pretrained,
                         require_all_grads=require_all_grads)


class ResNet50(nn.Module):
    def __init__(self, n_classes=1, pretrained=True, require_all_grads=False):
        super().__init__()
        if pretrained:
            self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.resnet = resnet50(weights=None)
        for param in self.resnet.parameters():
            param.requires_grad = require_all_grads
        last_layer_size = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(last_layer_size, n_classes)

    def forward(self, x):
        output = self.resnet(x)

        return output


class MobileNet(nn.Module):
    def __init__(self, n_classes=1, pretrained=True, require_all_grads=False):
        super().__init__()
        if pretrained:
            self.mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        else:
            self.mobilenet = mobilenet_v3_small()
        for param in self.mobilenet.parameters():
            param.requires_grad = require_all_grads
        last_layer_size = self.mobilenet.classifier._modules['3'].in_features
        self.mobilenet.classifier._modules['3'] = nn.Linear(last_layer_size, n_classes)

    def forward(self, x):
        output = self.mobilenet(x)
        return output