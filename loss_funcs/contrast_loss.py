import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import *
from sklearn.mixture import *
from geomstats.geometry.hypersphere import Hypersphere


def get_center(class_num, feat, labels):
    centers = torch.zeros(class_num, feat.shape[-1]).to(labels.device)
    for class_idx in range(class_num):
        # 获取当前类别的样本索引
        class_indices = (labels == class_idx).nonzero(as_tuple=True)[0]

        # 计算类中心，注意避免除零错误
        if len(class_indices) > 0:
            centers[class_idx] = torch.mean(feat[class_indices], dim=0)

    return centers


# 通过计算样本与中心之间的欧氏距离生成伪标签
def get_pseu_labels_base(feat, centers):
    pseu_labels = torch.zeros(feat.shape[0]).to(centers.device)
    for i, data in enumerate(feat):
        dist = torch.sum((data - centers) ** 2, dim=-1)
        pseu_labels[i] = torch.argmin(dist)

    return pseu_labels


# 通过计算样本与中心之间的测地线距离生成伪标签
def get_pseu_labels_hypersphere(feat, centers):
    dev = centers.device
    pseu_labels = torch.zeros(feat.shape[0]).to(dev)
    sphere = Hypersphere(dim=2)
    for i, data in enumerate(feat):
        # 使用测地线距离计算不同向量之间的距离
        dist = sphere.metric.dist(data.cpu().detach().numpy(), centers.cpu().detach().numpy())
        dist = torch.tensor(dist).to(dev)
        pseu_labels[i] = torch.argmin(dist)

    return pseu_labels


# 使用 K-Means聚类方法对数据进行聚类分析并生成伪标签
def get_pesu_labels_KMeans(feat, num_classes):
    kmeans = KMeans(n_clusters=num_classes)
    kmeans.fit(feat.cpu().detach().numpy())

    return torch.tensor(kmeans.labels_).to(feat.device)


def get_pesu_labels_DBSCAN(feat, num_classes):
    # 使用 DBSCAN聚类方法对数据进行聚类分析并生成伪标签 -- 会导致梯度爆炸，输出损失为 nan
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(feat.cpu().detach().numpy())

    return torch.tensor(dbscan.labels_).to(feat.device)


# 使用高斯混合模型对数据进行聚类分析并生成伪标签（Gaussian Mixture Models, GMM）
def get_pesu_labels_GMM(feat, num_classes):
    gmm = GaussianMixture(n_components=num_classes)
    gmm.fit(feat.cpu().detach().numpy())

    return torch.tensor(gmm.predict(feat.cpu().detach().numpy())).to(feat.device)


def inter_contrastive_loss(features, labels, temperature=0.5):
    # 计算余弦相似度
    similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)

    # 构建同类掩码矩阵
    labels = labels.unsqueeze(1)
    mask = torch.eq(labels, labels.T).float()

    # 计算分母
    exp_sim = torch.exp(similarity_matrix / temperature)
    exp_sim_sum = exp_sim.sum(dim=1, keepdim=True)

    # 计算对比损失
    log_prob = torch.log(torch.exp(similarity_matrix / temperature)) - torch.log(exp_sim_sum)
    loss = - (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
    loss = loss.mean()

    return loss


def target_to_center_loss(feat, pseu_labels, centers, num_classes, temperature=0.5):
    # 计算目标域样本和源域原型之间的余弦相似度
    similarity_matrix = F.cosine_similarity(feat.unsqueeze(1), centers.unsqueeze(0), dim=2)
    category_labels = torch.arange(0, num_classes)

    # 构建同类掩码矩阵
    mask = torch.eq(pseu_labels.unsqueeze(1), category_labels.unsqueeze(0)).float()

    # 计算分母
    exp_sim = torch.exp(similarity_matrix / temperature)
    exp_sim_sum = exp_sim.sum(dim=1, keepdim=True)

    # 计算对比损失
    log_prob = torch.log(torch.exp(similarity_matrix / temperature)) - torch.log(exp_sim_sum)
    loss = - (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
    loss = loss.mean()

    return loss


def contrastive_learning(src_feat, src_labels, tag_feat, pseu_labels, src_centers, num_classes):
    loss_src = inter_contrastive_loss(src_feat, src_labels)
    loss_tag = target_to_center_loss(tag_feat, pseu_labels, src_centers, num_classes=num_classes)
    loss = loss_src + loss_tag

    return loss


class ContrastLoss(nn.Module):
    def __init__(self, num_class):
        super(ContrastLoss, self).__init__()
        self.num_class = num_class

    def forward(self, src_feat, src_labels, tag_feat):
        src_centers = get_center(class_num=self.num_class, feat=src_feat, labels=src_labels)
        pesu_labels = get_pesu_labels_KMeans(feat=tag_feat, num_classes=self.num_class)
        loss = contrastive_learning(src_feat, src_labels, tag_feat, pesu_labels, src_centers, self.num_class)

        return loss


if __name__ == "__main__":
    x = torch.rand(16, 512)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    centers = get_center(class_num=4, feat=x, labels=labels)    # (4, 512)

    x_ = torch.rand(6, 512)
    pseu_labels = get_pseu_labels_base(feat=x_, centers=centers)
    print('pseu_labels:', pseu_labels.shape)
    print(pseu_labels)

    loss01 = inter_contrastive_loss(x, labels)
    print(loss01)

    loss02 = target_to_center_loss(x_, pseu_labels, centers, num_classes=4)
    print(loss02)

    loss = contrastive_learning(x, labels, x_, pseu_labels, centers)
    print(loss)
