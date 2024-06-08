import torch
import torch.nn as nn
import torchvision

from loss_funcs.grad_reverse import grad_reverse

import warnings
warnings.filterwarnings('ignore')

resnet = torchvision.models.resnet18(pretrained=True)


class FeatExtr(nn.Module):
    def __init__(self):
        super(FeatExtr, self).__init__()
        self.feat_mode = nn.Sequential(*list(resnet.children())[:-1],
                                       nn.Flatten(),
                                       )
        self.cls_mode = nn.Linear(resnet.fc.in_features, 2)
        self.dis_mode = nn.Linear(resnet.fc.in_features, 2)
        self.reverse = grad_reverse

    def forward(self, x):
        feat = self.feat_mode(x)

        return feat


class Classifier(nn.Module):
    def __init__(self, num_classes, grad_reverse=False):
        super(Classifier, self).__init__()
        self.cls_mode = nn.Linear(resnet.fc.in_features, num_classes)
        self.reverse = grad_reverse

    def forward(self, feat):
        if self.reverse:
            feat = grad_reverse(feat)
        res = self.cls_mode(feat)

        return res


if __name__ == '__main__':
    x = torch.rand(32, 3, 256, 256)
    feat_extr = FeatExtr()
    cls_mode01 = Classifier(num_classes=8)
    cls_mode02 = Classifier(num_classes=8)

    feat = feat_extr(x)
    res01 = cls_mode01(feat)
    res02 = cls_mode02(feat)

    print(feat.shape, res01.shape, res02.shape)
