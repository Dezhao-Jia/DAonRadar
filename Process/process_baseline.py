import copy
import random
import torch.cuda
import torch.backends.cudnn

import numpy as np
import torch.nn as nn

from Nets.resNet18 import FeatExtr, Classifier
from loss_funcs.contrast_loss import ContrastLoss
from loss_funcs.coral import CORAL
from Attention.baseline import CrossAttention
from Data.Datasets import get_Loader

import warnings

warnings.filterwarnings('ignore')


class Process:
    def __init__(self, args):
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_dis = nn.KLDivLoss(reduction="batchmean")
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.feat_mode = FeatExtr()
        self.cls_mode = Classifier(num_classes=2)
        # 使用 nn.Linear()构造的交叉注意力机制
        # self.aten_mode = CrossAttention(in_dim=512, out_dim=512, num_heads=1).to(self.device)
        # 使用 nn.Conv2d()构造的交叉注意力机制
        self.aten_mode = CrossAttention(channels=512).to(self.device)
        self.optim_feat = torch.optim.Adam(self.feat_mode.parameters(), lr=args.lr)
        self.optim_cls = torch.optim.Adam(self.cls_mode.parameters(), lr=args.lr)
        self.optim_aten = torch.optim.Adam(self.aten_mode.parameters(), lr=args.lr)
        self.loaders = get_Loader("Data", self.args.src_domain, self.args.tag_domain, self.args.batch_size)


    def torch_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    
    def train(self):
        net_dict, test_corr = self.do_train()
        # 保存断点信息
        path = "CheckPoint/backbone.pth"
        torch.save({'lr': self.args.lr, 'seed': self.args.seed, 'net_dict': net_dict, 'test_corr': test_corr}, path)
    
    
    def do_train(self):
        best_net = None
        corr_max = 0.0
        early_stop_epoch = 80
        remain_epoch = early_stop_epoch
        test_corr_max_list = []
    
        for epoch in range(self.args.max_epochs):
            self.feat_mode.train()
            self.cls_mode.train()
            # 当不使用交叉注意力机制时注释掉下一行即可
            self.aten_mode.train()
            s_feat, s_labels = [], []
            for dataset in self.loaders[0]:
                data, labels = dataset["image"].to(self.device), dataset["label"].to(self.device)
                feat = self.feat_mode(data)
                s_feat.append(feat)
                s_labels.append(labels)
    
            s_feat = torch.cat(s_feat, dim=0)
            s_labels = torch.cat(s_labels, dim=0)
    
            a_feat = []
            for dataset in self.loaders[1]:
                data, _ = dataset["image"].to(self.device), dataset["label"].to(self.device)
                feat = self.feat_mode(data)
                a_feat.append(feat)
    
            a_feat = torch.cat(a_feat, dim=0)
            print("feat shape :", s_feat.shape, s_labels.shape, a_feat.shape)
            # 当不使用交叉注意力机制时注释掉下一行即可
            s_res, a_res = self.aten_mode(s_feat, a_feat)
            s_out = self.cls_mode(s_res)
            a_out = self.cls_mode(a_res)
    
            loss_cls = self.loss_fn(s_out, s_labels)
            loss01 = self.loss_contrast(s_feat, s_labels, a_feat)
            # 使用CORAL度量不同分布之间的距离
            loss02 = CORAL(s_feat, a_feat)
            loss = loss_cls + self.alpha * loss01 + self.args.beta * loss02
            self.optim_feat.zero_grad()
            self.optim_cls.zero_grad()
            # 当不使用交叉注意力机制时注释掉下一行即可
            # self.optim_aten.zero_grad()
            loss.backward()
            self.optim_feat.step()
            self.optim_cls.step
            # 当不使用交叉注意力机制时注释掉下一行即可
            # self.optim_aten.step()
    
            src_corr, src_loss = self.evaluate_corr(self.loaders[0])
            tag_corr, tag_loss = self.evaluate_corr(self.loaders[-1])
            tag_corr_max_list.append(tag_corr)
    
            remain_epoch -= 1
            if tag_corr > corr_max:
                corr_max = tag_corr
                best_net = copy.deepcopy(self.net.state_dict())
                remain_epoch = early_stop_epoch
    
            if remain_epoch <= 0:
                break
    
            if epoch % 5 == 0:
                mes = 'epoch {:3d}, src_loss {:.5f}, src_corr {:.4f}, tag_loss {:.5f}, tag_corr {:.4f}' \
                    .format(epoch, src_loss, src_corr, tag_loss, tag_corr)
                print(mes)
    
        max_index = tag_corr_max_list.index(max(tag_corr_max_list))
        tag_corr = tag_corr_max_list[max_index]
    
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
