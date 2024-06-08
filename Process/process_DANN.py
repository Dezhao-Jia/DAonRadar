import os
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.backends.cudnn


from itertools import cycle
from Nets.DANN import Net
from Data.Datasets import get_Loader
from torch.utils.data import DataLoader


class Process:
    def __init__(self, args):
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.net = Net(num_classes=args.num_classes)
        self.loaders = get_Loader('../Data', args.src_domain, args.tag_domain. args.batch_size)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=args.lr)

    def torch_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def pre_training(self):
        path = self.args.check_point_dir + '/DANN/' + self.args.src_domain + '_' + self.args.tag_domain + '.pth'
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['best_net']).to(self.device)
        tag_corr, tag_loss = self.evaluate_corr(self.loaders[-1])

        sub_mess = 'sub {:2d}, tag_corr {:.4f}, tag_loss {:.4f}'.format(self.args.sub_id, tag_corr, tag_loss)
        print('=' * 50)
        print(sub_mess)
        print('=' * 50)

    def training(self):
        best_net, tag_corr = self.do_train()
        pre_path = self.args.check_point_dir + '/DANN'
        if not os.path.exists(pre_path):
            os.mkdir(pre_path)
        save_path = pre_path + '/' + self.args.src_domain + '_' + self.args.tag_domain + '.pth'
        if self.args.if_save:
            torch.save({'lr': self.args.lr, 'seed': self.args.seed, 'best_net': best_net, 'tag_corr': tag_corr},
                       save_path)

        domain_mess = 'src_domain {}, tag_domain {}, tag_corr {:.4f}' \
            .format(self.args.src_domain, self.args.tag_domain, tag_corr)
        print('=' * 50)
        print('sub_mess :')
        print(domain_mess)
        print('=' * 50)

    def do_train(self):
        best_net = None
        corr_max = 0.0
        early_stop_epoch = 100
        remain_epoch = early_stop_epoch
        tag_corr_list = []

        for epoch in range(self.args.max_epochs):
            self.net.train()
            src_cls_loss, src_dom_loss = 0.0, 0.0
            for datasets in self.loaders[0]:
                data, labels = datasets['image'].to(self.device), datasets['label'].to(self.device)
                src_feat, src_cls_res, src_dom_res = self.net(data)
                src_cls_loss += self.loss_fn(src_cls_res, labels)
                src_dom_loss += self.loss_fn(src_dom_loss, torch.zeros(data.shape[0]).long().to(self.device))

            aux_cls_loss, aux_dom_loss = 0.0, 0.0
            for datasets in self.loaders[1]:
                data, labels = datasets['image'].to(self.device), datasets['label'].to(self.device)
                aux_feat, aux_cls_res, aux_dom_res = self.net(data)
                aux_cls_loss += self.loss_fn(aux_cls_res, labels)
                aux_dom_loss += self.loss_fn(aux_dom_res, torch.ones(data.shape[0]).long().to(self.device))

            loss01 = src_cls_loss + aux_cls_loss
            loss02 = src_dom_loss + aux_dom_loss
            loss = loss01 + loss02
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            src_corr, src_loss = self.evaluate_corr(self.loaders[0])
            aux_corr, aux_loss = self.evaluate_corr(self.loaders[1])

            tag_corr, tag_loss = self.evaluate_corr(self.loaders[-1])
            tag_corr_list.append(tag_corr)
            if tag_corr > corr_max:
                corr_max = tag_corr
                best_net = copy.deepcopy(self.net.state_dict())
                remain_epoch = early_stop_epoch

            remain_epoch -= 1

            if epoch % 10 == 0:
                mess = "epoch {:3d}, src_loss {:.5f}, src_corr {:.4f}, aux_loss {:.5f}," \
                       "aux_corr {:.4f}, tag_loss {:.5f}, tag_corr {:.4f}" \
                    .format(epoch, src_loss, src_corr, aux_loss, aux_corr, tag_loss, tag_corr)
                print(mess)

            if remain_epoch <= 0:
                break

        # 选择验证准确率最高的一个 epoch对应的数据，打印并写入文件
        max_index = tag_corr_list.index(max(tag_corr_list))
        tag_corr = tag_corr_list[max_index]

        return best_net, tag_corr

    def evaluate_corr(self, data_iter):
        corr_sum, loss_sum = 0.0, 0.0
        size = 0
        with torch.no_grad():
            self.net.eval()
            for datasets in data_iter:
                data, labels = datasets['image'].to(self.device), datasets['label'].to(self.device)
                _, res, _ = self.net(data)
                loss = self.loss_fn(res, labels)
                _, pred = torch.max(res.data, dim=1)
                corr = pred.eq(labels.data).cpu().sum()
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
