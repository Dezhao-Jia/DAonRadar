import torch
import torch.nn as nn


class extrMode(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes):
        super(extrMode, self).__init__()
        self.mode = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=15, stride=1, padding=7),
                                       nn.BatchNorm2d(out_channels),
                                       nn.AvgPool2d(kernel_size=2, stride=2),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=1, padding=3),
                                       nn.BatchNorm2d(out_channels),
                                       nn.AvgPool2d(kernel_size=2, stride=2),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                       nn.AvgPool2d(kernel_size=2, stride=2),
                                       )

    def forward(self, x):
        o = self.mode(x)

        return o


class clsMode(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(clsMode, self).__init__()
        self.mode = nn.Sequential(nn.Conv2d(in_channels, n_classes, kernel_size=32, stride=1, padding=0),
                                  nn.Flatten(),
                                  nn.Sigmoid(),
                                  )

    def forward(self, x):
        o = self.mode(x)

        return o


if __name__ == '__main__':
    extr = extrMode(in_channels=3, out_channels=64, n_classes=8)
    print(extr)
    cls = clsMode(in_channels=64, n_classes=8)
    print(cls)

    x = torch.randn(1, 3, 256, 256)
    feat = extr(x)
    res = cls(feat)
    print(feat.shape, res.shape)    # torch.Size([1, 64, 32, 32]) torch.Size([1, 8])
