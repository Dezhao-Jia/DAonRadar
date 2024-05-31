import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, channels):
        super(CrossAttention, self).__init__()  # 定义线性层用于计算注意力权重
        self.q_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.k_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.v_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src_feat, tag_feat):  # 获取query、key、value
        b, c = src_feat.size()
        src_feat = src_feat.resize(b, c, 1, 1)
        tag_feat = tag_feat.resize(b, c, 1, 1)
        src_q = self.q_conv(src_feat).view(b, -1, 1).permute(0, 2, 1)
        src_k = self.k_conv(src_feat).view(b, -1, 1)
        src_v = self.v_conv(src_feat).view(b, -1, 1)

        tag_k = self.k_conv(tag_feat).view(b, -1, 1)
        tag_v = self.v_conv(tag_feat).view(b, -1, 1)

        # 计算注意力权重
        src_energy = torch.bmm(src_q, src_k)
        src_aten = self.softmax(src_energy)
        src_res = torch.bmm(src_v, src_aten.permute(0, 2, 1)).view(b, c)

        tag_energy = torch.bmm(src_q, tag_k)
        tag_aten = self.softmax(tag_energy)
        tag_res = torch.bmm(tag_v, tag_aten.permute(0, 2, 1)).view(b, c)

        return src_res, tag_res


# 示例用法
if __name__ == '__main__':
    b, c = 4, 256
    src_feat = torch.rand(b, c)
    tag_feat = torch.rand(b, c)
    cross_attention = CrossAttention(c)
    src_res, tag_res = cross_attention(src_feat, tag_feat)
    print(src_res.shape, tag_res.shape)
