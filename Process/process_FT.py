import copy
import random
import torch.cuda
import torch.backends.cudnn

import numpy as np
import torch.nn as nn

from Nets.resNet18 import FeatExtr, Classifier
from Data.Datasets import get_Loader

import warnings

warnings.filterwarnings('ignore')


class Process:
    def __init__(self, args):
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.feat_mode = FeatExtr().to(self.device)
        self.cls_mode = Classifier(num_classes=2).to(self.device)
        self.optim_feat = torch.optim.Adam(self.feat_mode.parameters(), lr=args.lr)
        self.optim_cls = torch.optim.Adam(self.cls_mode.parameters(), lr=args.lr)
        self.loaders = get_Loader("../Data", self.args.src_domain, self.args.tag_domain, self.args.batch_size)

    def torch_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def train(self):
        net_dict, test_corr = self.do_train()
        # path = "CheckPoint/backbone.pth"
        # torch.save({'lr': self.args.lr, 'seed': self.args.seed, 'net_dict': net_dict, 'test_corr': test_corr}, path)

    def do_train(self):
        best_net = None
        corr_max = 0.0
        early_stop_epoch = 80
        remain_epoch = early_stop_epoch
        tag_corr_max_list = []

        for epoch in range(self.args.max_epochs):
            self.feat_mode.train()
            self.cls_mode.train()
            for dataset in self.loaders[0]:
                data, labels = dataset["image"].to(self.device), dataset["label"].to(self.device)
                feat = self.feat_mode(data)
                res = self.cls_mode(feat)
                loss = self.loss_fn(res, labels)
                self.optim_feat.zero_grad()
                self.optim_cls.zero_grad()
                loss.backward()
                self.optim_feat.step()
                self.optim_cls.step()

            for dataset in self.loaders[1]:
                data, labels = dataset["image"].to(self.device), dataset["label"].to(self.device)
                feat = self.feat_mode(data)
                res = self.cls_mode(feat)
                loss = self.loss_fn(res, labels)
                self.optim_cls.zero_grad()
                loss.backward()
                self.optim_cls.step()

            src_corr, src_loss = self.evaluate_corr(self.loaders[0])
            aux_corr, aux_loss = self.evaluate_corr(self.loaders[1])
            tag_corr, tag_loss = self.evaluate_corr(self.loaders[-1])
            tag_corr_max_list.append(tag_corr)

            remain_epoch -= 1
            if tag_corr > corr_max:
                corr_max = tag_corr
                best_net = copy.deepcopy([self.feat_mode.state_dict(), self.cls_mode.state_dict()])
                remain_epoch = early_stop_epoch

            if remain_epoch <= 0:
                break

            if epoch % 1 == 0:
                mes = ('epoch {:3d}, src_loss {:.5f}, src_corr {:.4f}, aux_loss {:.5f}, '
                       'aux_corr {:.4f}, tag_loss {:.5f}, tag_corr {:.4f}') \
                    .format(epoch, src_loss, src_corr, aux_loss, aux_corr, tag_loss, tag_corr)
                print(mes)

        max_index = tag_corr_max_list.index(max(tag_corr_max_list))
        tag_corr = tag_corr_max_list[max_index]

        return best_net, tag_corr

    def evaluate_corr(self, data_iter):
        size = 0
        corr_sum, loss_sum = 0.0, 0.0
        with torch.no_grad():
            self.feat_mode.eval()
            self.cls_mode.eval()
            for dataset in data_iter:
                data, labels = dataset["image"].to(self.device), dataset["label"].to(self.device)
                feat = self.feat_mode(data)
                out = self.cls_mode(feat)
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
