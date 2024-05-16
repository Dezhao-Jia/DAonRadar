import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from loss_funcs.grad_reverse import grad_reverse


def reg(w1, w2):
    w1 = F.normalize(w1, dim=-1)
    w2 = F.normalize(w2, dim=-1)
    reg = torch.einsum("md,md->m", w1, w2)

    return (reg ** 2).sum(axis=-1).mean()


def compute_class_centers(features, labels, num_classes):
    """

    :param features: shape (nums, channels, points)
    :param labels: shape (nums)
    :param num_classes: int value
    :return:
    """
    class_centers = torch.zeros(num_classes, features.size(1), features.size(-1))  # 初始化类中心矩阵

    for class_idx in range(num_classes):
        # 获取当前类别的样本索引
        class_indices = (labels == class_idx).nonzero(as_tuple=True)[0]

        # 计算类中心，注意避免除零错误
        if len(class_indices) > 0:
            class_centers[class_idx] = torch.mean(features[class_indices], dim=0)

    return class_centers


# 定义中心损失函数
class CenterLoss(nn.Module):
    def __init__(self, num_classes):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, x, labels, centers):
        if len(x.shape) == 4:
            x = rearrange(x, 'n c h w -> n (c h) w')

        distance = 0.0
        for class_idx in range(self.num_classes):
            class_indices = (labels == class_idx).nonzero(as_tuple=True)[0]
            distance += torch.norm(x[class_indices] - centers[class_idx])

        return distance/self.num_classes


class CenterDisc(nn.Module):
    def __init__(self, num_classes):
        super(CenterDisc, self).__init__()
        self.num_classes = num_classes

    def forward(self, x, labels, centers):
        if len(x.shape) == 4:
            x = rearrange(x, 'n c h w -> n (c h) w')

        distance = 0.0
        num = 0
        for i in range(self.num_classes-1):
            for j in range(i+1, self.num_classes):
                distance += reg(centers[i], centers[j])
                num += 1

        return grad_reverse(distance/num)
