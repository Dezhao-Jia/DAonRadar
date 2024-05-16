import copy
import random
import torch.cuda
import torch.backends.cudnn

import numpy as np
import torch.nn as nn

from Nets.resNet18 import MyResnet18, FeatExtr, Classifier
from Data.Datasets import LoadData
from torch.utils.data import DataLoader

import warnings

warnings.filterwarnings('ignore')


class Process:
    def __init__(self, args):
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_dis = nn.KLDivLoss(reduction="batchmean")
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.datasets = LoadData().get_datasets()
        self.net = MyResnet18(num_classes=2)
        self.feat_mode = FeatExtr()
        self.cls_mode = Classifier(num_classes=2)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.train_iter = DataLoader(self.datasets[0], batch_size=args.batch_size, shuffle=True)
        self.test_iter = DataLoader(self.datasets[-1], batch_size=args.batch_size, shuffle=True)


def torch_seed(self):
    random.seed(self.args.seed)
    np.random.seed(self.args.seed)
    torch.manual_seed(self.args.seed)
    torch.cuda.manual_seed(self.args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(self):
    net_dict, test_corr = self.do_train()
    path = "CheckPoint/backbone.pth"
    torch.save({'lr': self.args.lr, 'seed': self.args.seed, 'net_dict': net_dict, 'test_corr': test_corr}, path)


def do_train(self):
    best_net = None
    corr_max = 0.0
    early_stop_epoch = 80
    remain_epoch = early_stop_epoch
    test_corr_max_list = []

    for epoch in range(self.args.max_epochs):
        self.net.train()
        t_feat = []
        for dataset in self.train_iter:
            data, labels = dataset["image"].to(self.device), dataset["label"].to(self.device)
            feat, out = self.net(data)
            t_feat.append(feat)
            loss = self.loss_fn(out, labels)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        t_feat = torch.cat(t_feat, dim=0)

        e_feat = []
        for dataset in self.test_iter:
            data, labels = dataset["image"].to(self.device), dataset["label"].to(self.device)
            feat, out = self.net(data)
            e_feat.append(feat)

        e_feat = torch.cat(e_feat, dim=0)
        print("feat shape :", t_feat.shape, e_feat.shape)

        kl_dis = self.loss_dis(t_feat.mean(dim=0), e_feat.mean(dim=0))
        print("KL Distance :", kl_dis)
        train_corr, train_loss = self.evaluate_corr(self.train_iter)
        test_corr, test_loss = self.evaluate_corr(self.test_iter)
        test_corr_max_list.append(test_corr)

        remain_epoch -= 1
        if test_corr > corr_max:
            corr_max = test_corr
            best_net = copy.deepcopy(self.net.state_dict())
            remain_epoch = early_stop_epoch

        if remain_epoch <= 0:
            break

        if epoch % 1 == 0:
            mes = 'epoch {:3d}, train_loss {:.5f}, train_corr {:.4f}, test_loss {:.5f}, test_corr {:.4f}' \
                .format(epoch, train_loss, train_corr, test_loss, test_corr)
            print(mes)

    max_index = test_corr_max_list.index(max(test_corr_max_list))
    tag_corr = test_corr_max_list[max_index]

    return best_net, tag_corr


def evaluate_corr(self, data_iter, iter_name="Source"):
    size = 0
    corr_sum, loss_sum = 0.0, 0.0
    with torch.no_grad():
        self.net.eval()
        for dataset in data_iter:
            data, labels = dataset["image"].to(self.device), dataset["label"].to(self.device)
            _, out = self.net(data)
            loss = self.loss_fn(out, labels)
            loss_sum += loss
            _, pred = torch.max(out.data, dim=1)
            corr = pred.eq(labels.data).cpu().sum()
            corr_sum += corr
            k = labels.data.shape[0]
            size += k
        corr_sum = corr_sum / size
        loss_sum = loss_sum / size

    return corr_sum, loss_sum
