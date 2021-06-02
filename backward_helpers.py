import torch
import numpy as np


def findNonZeros(x1, x2, thresh, _cython=False):

    n = len(x1)
    m = len(x2)
    nindex = np.zeros((n, 1))
    p = 0
    q = 0
    while q < m and p < n:
        if (x1[p] - x2[q]) < thresh:
            q += 1
        else:
            nindex[p] = q - 1 if q > 1 else 0
            p += 1
    if q == m:
        nindex[p] = q - 1 if q > 1 else 0
    if p != n:
        nindex[p:] = nindex[p]
    return torch.ByteTensor(nindex).cuda().squeeze(dim=-1)


def calCumSumLoss(Sx, Sy, nindex, DiN, tresh):
    n, m = Sx.shape[0], Sy.shape[0]
    Syx = DiN * Sy
    offset = 0
    w0 = 0
    z0 = 0
    deltaX = torch.zeros(n).cuda()
    DeltaX = torch.zeros(n).cuda()
    loc = 0
    for i in range(n):
        if nindex[i].item() != (offset - 1):
            end = nindex[i].item() + 1
            deltaX[i] = w0 + (DiN[offset:end].sum())
            DeltaX[i] = z0 + (Syx[offset:end].sum())
            offset = end
            w0 = deltaX[i]
            z0 = DeltaX[i]
            loc = i
        else:
            if nindex[i].item() == m - 1:
                break
            else:
                deltaX[i] = w0
                DeltaX[i] = z0
                loc = i
    if loc < n - 1:
        deltaX[loc:] = w0
        DeltaX[loc:] = z0

    return (torch.matmul((tresh - Sx), deltaX) + DeltaX.sum()).squeeze()


def calCumSumGradBP(Np, Nn, Ni, nindex, DiN):
    offset = 0
    grad_p = torch.zeros(Np).cuda()
    grad_n = torch.zeros(Nn).cuda()
    grad_0p = 0
    k = Np
    loc = 0
    for i in range(Np):
        if nindex[i].item() != (offset - 1):
            end = nindex[i].item() + 1
            grad_p[i] = grad_0p + (DiN[offset:end].sum())
            grad_n[offset:end] += k * DiN[offset:end]
            offset = end
            grad_0p = grad_p[i]
            k = -1
            loc = i
        else:
            if nindex[i].item() == Nn - 1:
                break
            else:
                grad_p[i] = grad_0p
                k -= 1
                loc = i
    if loc < Nn - 1:
        grad_p[loc:] = grad_0p

    return -grad_p, grad_n

class SqLossPerClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, Y, Yp, Yn, gamma):
        diff = pred - gamma * Y
        weight = Yp + Yn
        A = diff.mul(weight).dot(diff)
        B = (diff.dot(Yn)) * (Yp.dot(diff))
        ctx.save_for_backward(weight, diff, Yp, Yn)
        return 0.5 * A - B

    @staticmethod
    def backward(ctx, grad_output):
        # grad_x = grad_output.clone()
        weight, diff, Yp, Yn = ctx.saved_tensors
        grad = weight.mul(diff) - diff.dot(Yn) * Yp - diff.dot(Yp) * Yn
        return grad_output * grad, None, None, None, None


class ExpLossPerClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, Y, Yp, Yn, gamma):
        C1 = Yp * torch.exp(-gamma * pred)
        C2 = Yn * torch.exp(gamma * pred)
        C1s = C1.sum()
        C2s = C2.sum()
        # ctx.save_for_backward(C1, C2, C1s, C2s, gamma)
        grad = C1s * gamma * C2 - C2s * gamma * C1
        ctx.save_for_backward(grad)
        return C1s * C2s

    @staticmethod
    def backward(ctx, grad_output):
        # # grad_x = grad_output.clone()
        # C1, C2, C1s, C2s, gamma = ctx.saved_tensors
        # grad = C1s * gamma * C2 - C2s * gamma * C1
        grad = ctx.saved_tensors[0]
        return grad_output * grad, None, None, None, None


class HingeLossPerClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predi, Yi, Di, Ni, gamma):
        # Di = Di / Ni
        idx_p = Yi.eq(1)
        idx_n = Yi.ne(1)
        # predi = predi
        predi_p = predi[idx_p]
        predi_n = predi[idx_n]
        Di_n = Di[idx_n]
        idx_sort_p = predi_p.argsort(descending=True)
        idx_sort_n = predi_n.argsort(descending=True)
        predi_p = predi_p[idx_sort_p]
        predi_n = predi_n[idx_sort_n]
        Di_n = Di_n[idx_sort_n]
        nindex = findNonZeros(predi_p, predi_n, gamma)
        ctx.save_for_backward(idx_p, idx_n, idx_sort_p, idx_sort_n, nindex,
                              Di_n, Ni)
        return calCumSumLoss(predi_p, predi_n, nindex, Di_n, gamma)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_x = grad_output.clone()
        idx_p, idx_n, idx_sort_p, idx_sort_n,\
            nindex, Di_n, Ni = ctx.saved_tensors
        Np = idx_p.sum().item()
        Nn = idx_n.sum().item()
        grad_p, grad_n = calCumSumGradBP(Np, Nn, Ni, nindex, Di_n)
        grad_p_new, grad_n_new = torch.zeros(Np).cuda(), torch.zeros(Nn).cuda()
        grad_p_new[idx_sort_p] = grad_p
        grad_n_new[idx_sort_n] = grad_n
        grad_all = torch.zeros(Np + Nn).cuda()
        grad_all[idx_p] = grad_p_new
        grad_all[idx_n] = grad_n_new
        return (grad_output * grad_all), None, None, None, None
