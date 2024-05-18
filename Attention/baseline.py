import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, in_dim, k_dim, v_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.in_dim = in_dim
        self.func = baseAttention(in_dim, k_dim, v_dim, num_heads)

    def forward(self, src, tag):
        src_q, src_k, src_v = self.func(src)
        src_aten_matrix = torch.einsum('b d i, b d j -> b i j', src_q, src_k) * (self.in_dim ** -0.5)
        src_aten_matrix = src_aten_matrix.softmax(dim=-1)
        src_res = torch.einsum('b i j, b d j -> b d i', src_aten_matrix, src_v).view(src.shape[0], -1)

        _, tag_k, tag_v = self.func(tag)
        tag_aten_matrix = torch.einsum('b d i, b d j -> b i j', src_q, tag_k) * (self.in_dim ** -0.5)
        tag_aten_matrix = tag_aten_matrix.softmax(dim=-1)
        tag_res = torch.einsum('b i j, b d j -> b d i', tag_aten_matrix, tag_v).view(src.shape[0], -1)

        return src_res, tag_res
    
    
class baseAttention(nn.Module):
    def __init__(self, in_dim, k_dim, v_dim, num_heads=3):
        super(baseAttention, self).__init__()
        self.to_q = nn.Linear(in_dim, num_heads * k_dim, bias=False)
        self.to_k = nn.Linear(in_dim, num_heads * k_dim, bias=False)
        self.to_v = nn.Linear(in_dim, num_heads * v_dim, bias=False)

    def forward(self, feat):
        q = self.to_q(feat).view(feat.shape[0], 1, -1)
        k = self.to_k(feat).view(feat.shape[0], 1, -1)
        v = self.to_v(feat).view(feat.shape[0], 1, -1)

        return q, k, v


if __name__ == '__main__':
    x = torch.rand(16, 512)
    x_ = torch.rand(16, 512)
    aten_mode = CrossAttention(512, 64, 64, 4)
    res, res_ = aten_mode(x, x_)
    print(res.shape, res_.shape)
