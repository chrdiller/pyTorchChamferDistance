
import torch

from torch.utils.cpp_extension import load
cd = load(name="cd",
          sources=["chamfer_distance/chamfer_distance.cpp",
                   "chamfer_distance/chamfer_distance.cu"])

class ChamferDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        ctx.xyz1 = xyz1.contiguous()
        ctx.xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        ctx.idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        ctx.idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        if not xyz1.is_cuda:
            cd.forward(ctx.xyz1, ctx.xyz2, dist1, dist2, ctx.idx1, ctx.idx2)
        else:
            dist1 = dist1.cuda()
            dist2 = dist2.cuda()
            ctx.idx1 = ctx.idx1.cuda()
            ctx.idx2 = ctx.idx2.cuda()
            cd.forward_cuda(ctx.xyz1, ctx.xyz2, dist1, dist2, ctx.idx1, ctx.idx2)

        ctx.dist1 = dist1
        ctx.dist2 = dist2

        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(ctx.xyz1.size())
        gradxyz2 = torch.zeros(ctx.xyz2.size())

        if not graddist1.is_cuda:
            cd.backward(ctx.xyz1, ctx.xyz2, gradxyz1, gradxyz2, graddist1, graddist2, ctx.idx1, ctx.idx2)
        else:
            gradxyz1 = gradxyz1.cuda()
            gradxyz2 = gradxyz2.cuda()
            cd.backward_cuda(ctx.xyz1, ctx.xyz2, gradxyz1, gradxyz2, graddist1, graddist2, ctx.idx1, ctx.idx2)

        return gradxyz1, gradxyz2


class ChamferDistance(torch.nn.Module):
    def forward(self, xyz1, xyz2):
        return ChamferDistanceFunction.apply(xyz1, xyz2)
