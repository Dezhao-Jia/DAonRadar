import torch
import torch.nn as nn
import torch.nn.functional as F


def get_center(class_num, feat, labels):
    centers = torch.zeros(class_num, feat.shape[-1]).to(labels.device)
    for class_idx in range(class_num):
        # 获取当前类别的样本索引
        class_indices = (labels == class_idx).nonzero(as_tuple=True)[0]

        # 计算类中心，注意避免除零错误
        if len(class_indices) > 0:
            centers[class_idx] = torch.mean(feat[class_indices], dim=0)

    return centers


def get_pseu_labels(feat, centers):
    pseu_labels = torch.zeros(feat.shape[0]).to(centers.device)
    for i, data in enumerate(feat):
        dist = torch.sum((data - centers) ** 2, dim=-1)
        pseu_labels[i] = torch.argmin(dist)

    return pseu_labels


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
        pesu_labels = get_pseu_labels(feat=tag_feat, centers=src_centers)
        loss = contrastive_learning(src_feat, src_labels, tag_feat, pesu_labels, src_centers, self.num_class)

        return loss


if __name__ == "__main__":
    x = torch.rand(16, 512)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    centers = get_center(class_num=4, feat=x, labels=labels)    # (4, 512)

    x_ = torch.rand(6, 512)
    pseu_labels = get_pseu_labels(feat=x_, centers=centers)
    print('pseu_labels:', pseu_labels.shape)
    print(pseu_labels)

    loss01 = inter_contrastive_loss(x, labels)
    print(loss01)

    loss02 = target_to_center_loss(x_, pseu_labels, centers, num_classes=4)
    print(loss02)

    loss = contrastive_learning(x, labels, x_, pseu_labels, centers)
    print(loss)
