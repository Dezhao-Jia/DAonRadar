import os
import copy
import torch
import random
import torch.backends.cudnn

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Data.Datasets import get_Loader
from Nets.MCD import FeatExtr, Classifier


# 衡量两个结果之间的距离，使用均值作为差距
def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1, dim=0) - F.softmax(out2, dim=0)))


class Process:
    def __init__(self, args):
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaders = get_Loader('Data', args.src_domain, args.tag_domain, args.batch_size)
        self.feat_extr = FeatExtr().to(self.device)
        self.feat_optim = optim.Adam(self.feat_extr.parameters(), lr=args.lr)
        self.cls_mode01 = Classifier(num_classes=args.num_classes).to(self.device)
        self.cls_optim01 = optim.Adam(self.cls_mode01.parameters(), lr=args.lr)
        self.cls_mode02 = Classifier(num_classes=args.num_classes).to(self.device)
        self.cls_optim02 = optim.Adam(self.cls_mode02.parameters(), lr=args.lr)

    def torch_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def pre_training(self):
        path = self.args.check_point_dir + '/MCD/' + self.args.src_domain + '_' + self.args.tag_domain + '.pth'
        checkpoint = torch.load(path)
        self.feat_extr.load_state_dict(checkpoint['best_feat_extr'])
        self.cls_mode01.load_state_dict(checkpoint['best_cls_mode01'])
        self.cls_mode02.load_state_dict(checkpoint['best_cls_mode02'])

        tag_corr, tag_loss = self.evaluate_corr(self.loaders[-1])
        info_mess = 'src_domain {}, tag_domain {}, tag_corr_max {:.4f}, tag_corr_mean {:.4f}' \
            .format(self.args.src_domain, self.args.tag_domain, tag_corr, tag_loss)
        print('=' * 50)
        print(info_mess)
        print('=' * 50)

    def training(self):
        print('='*10, "src_domain {}, tag_domain {}".format(self.args.src_domain, self.args.tag_domain), '='*10)
        print('='*10, "Step A : Train mode with source domain datasets !", '='*10)
        net_list = self.stepA()

        print('='*10, "Loading state dict to the part of network !", '='*10)
        self.feat_extr.load_state_dict(net_list[0])
        self.cls_mode01.load_state_dict(net_list[1])
        self.cls_mode02.load_state_dict(net_list[2])

        print('='*10, "Step B : Transfer learning training with part of target domain datasets !", '='*10)
        net_list, info_list = self.do_train()

        pre_path = self.args.check_point_dir + '/MCD'
        if not os.path.exists(pre_path):
            os.mkdir(pre_path)
        save_path = pre_path + '/' + self.args.src_domain + '_' + self.args.tag_domain + '.pth'
        if self.args.if_save:
            torch.save({'lr': self.args.lr, 'seed': self.args.seed,
                        'best_feat_extr': net_list[0], 'best_cls_mode01': net_list[1], 'best_cls_mode02': net_list[-1],
                        'tag_loss': info_list[0], 'tag_corr': info_list[-1],
                        }, save_path)

    def stepA(self):
        """
        stepA: train the feature extractor and two classifiers
        """
        corr_max = 0.0
        best_feat_extr = None
        best_cls_mode01 = None
        best_cls_mode02 = None
        for epoch in range(200):
            self.feat_extr.train()
            self.cls_mode01.train()
            self.cls_mode02.train()
            size = 0
            early_stop_epoch = 20
            corr_sum, loss_sum = 0.0, 0.0
            remain_epoch = early_stop_epoch
            for datasets in self.loaders[0]:
                data, labels = datasets['image'].to(self.device), datasets['label'].to(self.device)
                feat = self.feat_extr(data)
                res01 = self.cls_mode01(feat)
                res02 = self.cls_mode02(feat)
                res = res01 + res02
                _, pred = torch.max(res.data, dim=1)
                corr = pred.eq(labels.data).cpu().sum()
                corr_sum += corr

                loss_s1 = self.loss_fn(res01, labels).item()
                loss_s2 = self.loss_fn(res02, labels).item()
                loss = loss_s1 + loss_s2
                loss_sum += loss
                k = labels.data.shape[0]
                size += k
                self.feat_optim.zero_grad()
                self.cls_optim01.zero_grad()
                self.cls_optim02.zero_grad()
                loss.backward()
                self.feat_optim.step()
                self.cls_optim01.step()
                self.cls_optim02.step()
            corr_sum = corr_sum / size
            loss_sum = loss_sum / len(self.loaders[0])
            if corr_sum > corr_max:
                corr_max = corr_sum
                best_feat_extr = copy.deepcopy(self.feat_extr.state_dict())
                best_cls_mode01 = copy.deepcopy(self.cls_mode01.state_dict())
                best_cls_mode02 = copy.deepcopy(self.cls_mode02.state_dict())
                remain_epoch = early_stop_epoch

            remain_epoch -= 1

            if epoch % 10 == 0:
                print('epoch {:2d}, loss {:.5f}, corr {:.4f}'.format(epoch, loss_sum, corr_sum))

            if remain_epoch <= 0:
                break

        return [best_feat_extr, best_cls_mode01, best_cls_mode02]

    def do_train(self):
        """
        do stepB and stepC
        """
        best_feat_extr = None
        best_cls_mode01 = None
        best_cls_mode02 = None
        corr_max = 0.0
        early_stop_epoch = 100
        remain_epoch = early_stop_epoch

        tag_corr_list = []
        tag_loss_list = []
        for epoch in range(self.args.max_epochs):
            loss_cls, loss_dis = 0.0, 0.0
            for datasets in self.loaders[0]:
                data, labels = datasets['image'].to(self.device), datasets['label'].to(self.device)
                feat = self.feat_extr(data)
                res01 = self.cls_mode01(feat)
                res02 = self.cls_mode02(feat)
                loss01 = self.loss_fn(res01, labels).item()
                loss02 = self.loss_fn(res02, labels).item()
                loss_cls = loss01 + loss02

            for datasets in self.loaders[1]:
                data = datasets['image'].to(self.device)
                feat = self.feat_extr(data)
                res01 = self.cls_mode01(feat)
                res02 = self.cls_mode02(feat)
                loss_dis = discrepancy(res01, res02)

            if epoch % 2 == 0:
                self.cls_optim01.zero_grad()
                self.cls_optim02.zero_grad()
                loss = loss_cls - loss_dis
                loss.backward()
                self.cls_optim01.step()
                self.cls_optim02.step()
            else:
                self.feat_optim.zero_grad()
                loss = loss_dis
                loss.backward()
                self.feat_optim.step()

            tag_corr, tag_loss = self.evaluate_corr(self.loaders[-1])
            tag_corr_list.append(tag_corr)
            tag_loss_list.append(tag_loss)
            remain_epoch -= 1
            if tag_corr > corr_max:
                corr_max = tag_corr
                best_feat_extr = copy.deepcopy(self.feat_extr.state_dict())
                best_cls_mode01 = copy.deepcopy(self.cls_mode01.state_dict())
                best_cls_mode02 = copy.deepcopy(self.cls_mode02.state_dict())
                remain_epoch = early_stop_epoch

            if remain_epoch <= 0:
                break

            if epoch % 10 == 0:
                mess = 'epoch {:3d}, tag_loss {:.5f}, tag_corr {:.4f}'.format(epoch, tag_loss, tag_corr)
                print(mess)

        max_index = tag_corr_list.index(max(tag_corr_list))
        tag_corr = tag_corr_list[max_index]
        tag_loss = tag_loss_list[max_index]

        return [best_feat_extr, best_cls_mode01, best_cls_mode02], [tag_loss, tag_corr]

    def evaluate_corr(self, data_iter):
        corr_sum, loss_sum = 0.0, 0.0
        size = 0
        with torch.no_grad():
            self.feat_extr.eval()
            self.cls_mode01.eval()
            self.cls_mode02.eval()
            for datasets in data_iter:
                data, labels = datasets['image'].to(self.device), datasets['label'].to(self.device)
                feat = self.feat_extr(data)
                res01 = self.cls_mode01(feat)
                res02 = self.cls_mode02(feat)
                loss = self.loss_fn(res02, labels).item()
                out_esm = res01 + res02
                _, pred_esm = torch.max(out_esm.data, dim=1)
                corr = pred_esm.eq(labels.data).cpu().sum()
                corr_sum += corr
                loss_sum += loss
                k = labels.data.shape[0]
                size += k
            corr_sum = corr_sum / size
            loss_sum = loss_sum / len(data_iter)

        return corr_sum, loss_sum

    def running(self):
        self.torch_seed()
        if self.args.pretrain:
            self.pre_training()
        else:
            self.training()
