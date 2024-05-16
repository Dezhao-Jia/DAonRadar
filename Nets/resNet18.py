import torch
import torchvision

import torch.nn as nn

import warnings
warnings.filterwarnings('ignore')

resnet = torchvision.models.resnet18(pretrained=True)


class MyResnet18(nn.Module):
    def __init__(self, num_classes):
        super(MyResnet18, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.feat_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        self.classifier = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        feat = self.feat_extractor(x)
        res = self.classifier(feat.view(feat.size(0), -1))

        return feat, res


class FeatExtr(nn.Module):
    def __init__(self):
        super(FeatExtr, self).__init__()
        self.mode = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        feat = self.mode(x)

        return feat


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.mode = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        res = self.mode(x.view(x.size(0), -1))

        return res


if __name__ == '__main__':
    net = MyResnet18(num_classes=3)
    x = torch.rand(32, 3, 256, 256)

    feat, out = net(x)
    print(feat.shape, out.shape)    # (32, 512, 1, 1) (32, 3)
