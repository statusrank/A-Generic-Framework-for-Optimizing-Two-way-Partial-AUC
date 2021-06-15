from token import NT_OFFSET
from numpy.core.numeric import require
from numpy.lib.polynomial import RankWarning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from abc import abstractmethod
import numpy as np 

from abc import abstractmethod
import pdb

class BaseLoss(nn.Module):
    def __init__(self):

        super(BaseLoss, self).__init__()

    def preprocess_inputs(self, logit, target):
        if len(torch.unique(target)) > 2:
            raise RuntimeError(
                'Class Number Greater than Two: only binary class problems are supported')
        if len(torch.unique(target)) == 1:
            raise RuntimeError(
                'One Class Missing: AUC evaluation requires both classes in the mini-batch')
        if len(target.shape) == 2 and target.shape[1] > 1:
            raise RuntimeError('Label Output Should be a Vector')
       
        if len(logit.shape) == 1:
            logit = logit.view(-1, 1)
        if logit.shape[1] > 1:
            pred = torch.softmax(logit, 1)[:, 1]
        else:
            pred = torch.sigmoid(logit)
        
        pred = pred.squeeze()

        return pred, target

    def forward(self, logit, target, **kwargs):
        pass

class CELoss(BaseLoss):
    def __init__(self, reduction='mean', **kwargs):
        super(CELoss, self).__init__() 

        self.reduction = reduction
    
    def forward(self, logit, target, epoch=0):

        """

        :param pred: the predicted score vector
        :param target: the GT label vector

        """

        np = target.eq(1).sum().float()
        nn = target.ne(1).sum().float()
        sam_weight = 1.0 * torch.zeros_like(target)
        sam_weight[target.eq(1)] = 1
        sam_weight[target.ne(1)] = np/nn

        return F.binary_cross_entropy_with_logits(logit.view(-1), target.float(), weight= sam_weight)


class FOCALLoss(BaseLoss):
    def __init__(self, focal_gamma, alpha=None, size_average=True, **kwargs):
       
        super(FOCALLoss, self).__init__()
        num_class = 2
        if alpha is None:
            self.alpha = Variable(torch.ones(num_class, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.alpha = self.alpha.cuda()

        self.gamma = focal_gamma
        self.num_class = num_class
        self.size_average = size_average

    def forward(self, logit, target, epoch=0):
        inputs = logit

        N = inputs.size(0)
        C = max(2, inputs.size(1))

        if inputs.shape[1] == 1:
            P = torch.sigmoid(inputs)
            P = torch.cat([1-P, P], 1)
        else:
            P = torch.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = target.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        assert P.shape == (N, C), 'only care the class-1 channel'
        assert P.shape == class_mask.shape

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(inputs.device)

        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class CBFOCALLoss(BaseLoss):
    def __init__(self, cls_num_list, cb_beta, focal_gamma, **kwargs):
        super(CBFOCALLoss, self).__init__()
        num_class = 2
        if cls_num_list is None:
            return
        self.beta = cb_beta
        self.cls_num_list = cls_num_list
        n_y = torch.Tensor(self.cls_num_list).float().cuda()
        weight = (1 - self.beta) / (1 - self.beta**n_y)
        weight = weight / torch.sum(weight) * num_class
        weight = weight.view(num_class, 1)
        self.focal = FOCALLoss(alpha=weight, focal_gamma=focal_gamma)

    def forward(self, logit, target, epoch=0):
        return self.focal(logit, target)

class CBCELoss(BaseLoss):
    def __init__(self, num_class, cls_num_list, cb_beta, **kwargs):
        super(CBCELoss, self).__init__()
        if cls_num_list is None:
            return
        self.beta = cb_beta
        self.cls_num_list = cls_num_list
        n_y = torch.Tensor(self.cls_num_list).float().cuda()
        self.weight = (1 - self.beta) / (1 - self.beta**n_y)
        self.weight = self.weight / torch.sum(self.weight) * num_class

    def forward(self, logit, target, epoch=0):
        if logit.shape[1] == 1:
            weight = torch.index_select(self.weight, 0, target)
            return F.binary_cross_entropy_with_logits(logit.view(-1), target.float(), weight=weight)
        else:
            return F.cross_entropy(logit, target, weight=self.weight)


class SquareAUCLoss(BaseLoss):
    def __init__(self, gamma=1, useManualBackprop=False, **kwargs):
        super(SquareAUCLoss, self).__init__()
        self.gamma = gamma
        self.reduction = 'mean'

    def forward(self, logit, target, epoch=0):
        pred, target = self.preprocess_inputs(logit, target)
        pred_p = pred[target.eq(1)]
        pred_n = pred[target.ne(1)]

        n_plus, n_minus = len(pred_p), len(pred_n)

        pred_p = pred_p.unsqueeze(1).expand(n_plus, n_minus)
        pred_n = torch.reshape(pred_n, (1, n_minus))

        loss = (self.gamma + pred_n - pred_p) ** 2
        
        return loss.mean() if self.reduction == 'mean' else loss.sum()

class TPAUCLoss(BaseLoss):
    def __init__(self, gamma, 
                 epoch_to_paced,
                 re_scheme,
                 num_class = 2, 
                 norm=False,
                 reduction='mean', **kwargs):
        
        super(TPAUCLoss,self).__init__()
        self.gamma = gamma
        
        assert num_class == 2, 'must be binary classification task'
        self.num_class = num_class

        self.reduction = reduction

        # adopt re_weight func after epoch_to_paced epoch....
        self.epoch_to_paced = epoch_to_paced

        if re_scheme not in ["Exp", 'Poly']:
            raise ValueError

        self.re_scheme = re_scheme

        self.norm = norm

    def forward(self, logit, target, epoch=0):
        pred, target = self.preprocess_inputs(logit, target)

        pred_p = pred[target.eq(1)]
        pred_n = pred[target.ne(1)]

        n_plus, n_minus = len(pred_p), len(pred_n)

        if epoch >= self.epoch_to_paced:
            weight = self.re_weight(pred_p, pred_n)
        else:
            weight = torch.ones(n_plus, n_minus)
        assert weight.shape == (n_plus, n_minus)

        if pred.is_cuda and not weight.is_cuda:
            weight = weight.cuda()

        pred_p = pred_p.unsqueeze(1).expand(n_plus, n_minus)
        pred_n = torch.reshape(pred_n, (1, n_minus))

        Delta = (1 + pred_n - pred_p) ** 2

        loss =  Delta * weight
        return loss.mean() if self.reduction == 'mean' else loss.sum()

    def re_weight(self, pred_p, pred_n, eps=1e-6):
        '''
        return:
            must be the (len(pred_p), len(pred_n)) matrix 
                    for element-wise multiplication
        '''

        if self.re_scheme == 'Poly':
            
            col_pred_p = torch.pow((1 - pred_p + eps), self.gamma)
            row_pred_n = torch.pow(pred_n + eps, self.gamma)

        elif self.re_scheme == 'Exp':

            col_pred_p = 1 - torch.exp(- self.gamma * (1 - pred_p))
            row_pred_n = 1 - torch.exp(- self.gamma * pred_n)
        else:
            raise ValueError

        return torch.mm(col_pred_p.unsqueeze_(1), row_pred_n.unsqueeze_(0))


def get_loss(config, num_class, cls_num_list, freq_info=None):
    loss_type = config['loss_type']
    config = config['loss_params']
    config['num_class'] = num_class
    config['cls_num_list'] = cls_num_list
    config['freq_info'] = freq_info

    if loss_type not in globals().keys():
        if loss_type + 'Loss' in globals().keys():
            loss_type = loss_type + 'Loss'
        else:
            raise NotImplementedError('Unknown loss_type: %s'%loss_type)

    return globals()[loss_type](**config)

if __name__ == '__main__':
    cri = TPAUCLoss(gamma=0.1, epoch_to_paced=0, re_scheme='Poly')

    pred = torch.rand(10)
    target = torch.rand(10) 
    target[target >= 0.5] = 1
    target[target < 0.5] = 0

    # print(pred)
    # print(target)

    loss = cri(pred, target, 1)

    print(loss)

