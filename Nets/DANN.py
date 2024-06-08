import torch
import torchvision
import torch.nn as nn

from loss_funcs.grad_reverse import grad_reverse

import warnings

warnings.filterwarnings('ignore')

resnet = torchvision.models.resnet18(pretrained=True)


class Net(nn.Module):
    def __init__(self, num_classes, num_domains=2, grad_reverse=True):
        super(Net, self).__init__()
        self.feat_mode = nn.Sequential(*list(resnet.children())[:-1],
                                       nn.Flatten(),
                                       )

        self.cls_mode = nn.Linear(resnet.fc.in_features, num_classes)

        self.dis_mode = nn.Linear(resnet.fc.in_features, num_domains)
        self.reverse = grad_reverse

    def forward(self, x):
        feat = self.feat_mode(x)
        cls_res = self.cls_mode(feat)

        if self.reverse:
            feat = grad_reverse(feat, lambd=0.2)
        dis_res = self.dis_mode(feat)

        return feat, cls_res, dis_res


if __name__ == '__main__':
    x = torch.rand(32, 3, 256, 256)
    net = Net(num_classes=8)
    feat, cls_res, dis_res = net(x)

    print(feat.shape, cls_res.shape, dis_res.shape)
