import torch
from einops import rearrange


def CORAL(source, target):
    if len(source.shape) == 4:
        source = rearrange(source, 'n c h w -> n (c h w)')
        target = rearrange(target, 'n c h w -> n (c h w)')

    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]

    xm = torch.mean(source, dim=0, keepdim=True) - source
    xc = xm.t() @ xm / (ns - 1)

    xmt = torch.mean(target, dim=0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)

    loss = torch.mul((xc - xct), (xc - xct))
    loss = torch.sum(loss) / (4*d*d)

    return loss


if __name__ == '__main__':
    src_feat = torch.rand(4, 512)
    tag_feat = torch.rand(4, 512)
    print(CORAL(src_feat, tag_feat))
