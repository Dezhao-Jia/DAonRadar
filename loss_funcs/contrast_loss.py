import torch

from loss_funcs.center_loss import CenterLoss, CenterDisc


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
    pseu_labels = torch.zeros(feat.shape[0]).to(labels.device)
    for i, data in enumerate(feat):
        dist = torch.sum((data - centers) ** 2, dim=-1)
        pseu_labels[i] = torch.argmin(dist)

    return pseu_labels


def contrastive_learning(src_feat, src_labels, tag_feat, pseu_labels, src_centers, tag_centers):
    loss_func1 = CenterLoss(num_classes=4)
    loss_func2 = CenterDisc(num_classes=4)
    src_loss = loss_func1(src_feat, src_labels, src_centers) + loss_func2(tag_feat, pseu_labels, tag_centers)
    tag_loss = loss_func1(tag_feat, pseu_labels, tag_centers) + loss_func2(src_feat, src_labels, src_centers)
    dom_dis = torch.mean(torch.abs(src_centers - tag_centers))
    loss = src_loss + tag_loss + dom_dis

    return loss


if __name__ == "__main__":
    x = torch.rand(16, 512)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    centers = get_center(class_num=4, feat=x, labels=labels)
    print(centers.shape)

    x_ = torch.rand(6, 512)
    pseu_labels = get_pseu_labels(feat=x_, centers=centers)
    print('pseu_labels:', pseu_labels.shape)
    print(pseu_labels)
