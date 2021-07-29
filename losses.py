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

