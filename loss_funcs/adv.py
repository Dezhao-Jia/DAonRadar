import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Function

from neural_nets.MLP import MLP


class LambdaSheduler(nn.Module):
    def __init__(self, gamma=1.0, max_iter=1000):
        super(LambdaSheduler, self).__init__()
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0

    def lamb(self):
        p = self.curr_iter / self.max_iter
        lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1

        return lamb

    def step(self):
        self.curr_iter = min(self.curr_iter+1, self.max_iter)


class AdversarialLoss(nn.Module):
    def __init__(self, in_features, gamma=1.0, max_iter=1000, use_lambda_scheduler=True):
        super(AdversarialLoss, self).__init__()
        self.dis = MLP(in_features, 1)
        self.use_lambda_scheduler = use_lambda_scheduler
        if self.use_lambda_scheduler:
            self.lambda_scheduler = LambdaSheduler(gamma, max_iter)

    def forward(self, source, target):
        lamb = 1.0
        if self.use_lambda_scheduler:
            lamb = self.lambda_scheduler.lamb()
            self.lambda_scheduler.step()
        source_loss = self.get_adversial_result(source, True, lamb)
        target_loss = self.get_adversial_result(target, False, lamb)
        adv_loss = 0.5 * (source_loss + target_loss)

        return adv_loss

    def get_adversial_result(self, x, source=True, lamb=1.0):
        x = ReverseLayerF(x, lamb)
        dis_pred = self.dis(x)
        device = dis_pred.device
        if source:
            dis_label = torch.ones(len(x), 1).long()
        else:
            dis_label = torch.zeros(len(x), 1).long()

        loss_fn = nn.BCELoss()
        loss_adv = loss_fn(dis_pred, dis_label.float().to(device))

        return loss_adv


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
