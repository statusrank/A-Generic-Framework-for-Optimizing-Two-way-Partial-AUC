import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import sys
import json
from easydict import EasyDict as edict
import time
import os
import os.path as osp
from tqdm import tqdm

import models
import utils
from dataloaders import get_datasets, get_data_loaders, StratifiedSampler
from metric import p2AUC, paucZero, argBottomk, argTopk, devec, vec
from losses import get_loss
from models import generate_net
from utils import MyLog
from itertools import product
from torch.utils.data import DataLoader
import copy
from collections import defaultdict
import pdb

def test(model, loader, class_dim, params):
    model.eval()
    preds = []
    ty = []
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.cuda().float(), by.cpu().numpy()
            # pdb.set_trace()
            output = model(bx)
            if output.shape[1] == 1:
                yp = torch.sigmoid(output).cpu().numpy()
            else:
                yp = torch.softmax(output, 1)[:, 1].cpu().numpy()
            preds.append(yp)
            ty.append(by)
    pred = np.concatenate(preds, axis=0)

    ty = np.concatenate(ty, axis=0)

    p2auc = p2AUC(ty, pred, params.alpha, params.beta)
    return p2auc

def train(args, use_test=True, idx=0):
    

    logger_base = osp.join(args.training.save_path, args.training.loss_type)

    if "re_scheme" in args.training.loss_params.keys():
        logger_base = osp.join(logger_base, args.training.loss_params.re_scheme)

    logger_base = osp.join(logger_base, 'class_' + str(idx))

    if not osp.exists(logger_base):
        os.makedirs(logger_base)

    base_model = "_".join(["metric_" + str(key) + "_" + str(val) 
                        for key, val in args.training.metric_params.items()]) 
    best_model_path = osp.join(logger_base, base_model + '.pth')

    logger_path = 'lr_' + str(args.training.lr) + '_' + base_model
    
    if "alpha" in args.training.loss_params.keys():
        logger_path = logger_path + '_' + \
                      'train_alpha_' + str(args.training.loss_params.alpha) + \
                       '_beta_' + str(args.training.loss_params.beta)
    if "epoch_to_paced" in args.training.keys():
        logger_path += '_epo_' + str(args.training.epoch_to_paced)

    if "epoch_num" in args.training.keys():
        logger_path += '_epoch_num_' + str(args.training.epoch_num)

    if "epoch_to_paced" in args.training.loss_params.keys():
        logger_path += '_epo_' + str(args.training.loss_params.epoch_to_paced)
    
    if "gamma" in args.training.loss_params.keys():
        logger_path += '_gamma_' + str(args.training.loss_params.gamma)

    if "norm" in args.training.loss_params.keys():
        logger_path += '_norm_' + str(args.training.loss_params.norm)

    logger_path = logger_path + '.log'
 
    logger_filename = osp.join(logger_base, logger_path) 
    logger = MyLog(logger_filename,name = logger_filename)

    params_dicts = '\n'.join([str(k) + '=' + str(v) \
                            for k, v in sorted(vars(args.training).items(), key=lambda x: x[0])])
    
    # record the params
    logger.info(params_dicts)

    utils.setup_seed(args.training.seed)

    random_state = np.random.RandomState(args.training.seed)

    train_set, val_set, test_set = get_datasets(args.dataset)

    train_loader, val_loader, test_loader = get_data_loaders(
        train_set, val_set, test_set,
        args.training.train_batch_size,
        args.training.test_batch_size,
        args.training.num_workers,
        args.dataset.sampler.rpos,
        args.dataset.sampler.rneg,
        random_state)

    model = generate_net(args.model)
    model.cuda()

    criterion = get_loss(args.training,
                         train_set.get_num_classes(),
                         train_set.get_cls_num_list(),
                         train_set.get_freq_info())

    optmizer = torch.optim.SGD(model.parameters(),
                               lr=args.training.lr,
                               weight_decay=args.training.weight_decay,
                               momentum=args.training.momentum,
                               nesterov=args.training.nesterov)

    scheduler = lr_scheduler.ExponentialLR(optmizer,
                                           args.training.lr_decay_rate)

    global_step = 0
    epoch_num = args.get('real_epoch_num',
                           args.training.epoch_num)

    best_p2auc = 0
    best_pzero = 0

    test_p2auc = 0
    test_pzero = 0

    p2auc = test(model, val_loader, train_set.get_class_dim(), args.training.metric_params)
    logger.info('\nEvaluating validation set...')
    logger.info('p2auc:  %.4f previous best p2auc:  %.4f'%(p2auc, best_p2auc))
    best_p2auc = p2auc

    for epoch in range(epoch_num):
        logger.info('\n========> Epoch %3d: '% epoch) 
        for it, (bx, by) in enumerate(train_loader):
            bx, by = bx.cuda().float(), by.cuda().long()
            
            model.train()
            pred = model(bx)
            loss = criterion(pred, by.view(-1), epoch)
        
            optmizer.zero_grad()
            loss.backward()
            optmizer.step()
        
            global_step += 1

            if it % 100 == 0 or it == len(train_loader) - 1:
                logger.info('Iter %4d/%4d:  loss: %.4f'%(it, len(train_loader), loss.item()))
        
        logger.info('\nEvaluating training set...')
        p2auc = test(model, train_loader, train_set.get_class_dim(), args.training.metric_params)
        logger.info('p2auc:  %.4f'%(p2auc))

        logger.info('\nEvaluating validation set...')
        p2auc= test(model, val_loader, train_set.get_class_dim(), args.training.metric_params)
        logger.info('p2auc:  %.4f  previous best p2auc: %.4f'%(p2auc, best_p2auc))
        
        # save best model        
        if p2auc > best_p2auc:

            best_p2auc = p2auc

            logger.info('\nEvaluating test set...')
            test_p2auc = test(model, test_loader, train_set.get_class_dim(), args.training.metric_params)
            torch.save(model.state_dict(), best_model_path)
            logger.info('test_p2auc:  %.4f '%(test_p2auc))

        if (epoch + 1) % args.training.lr_decay_epochs == 0:
            scheduler.step()

    logger.info('\nFinal results (test set) ====> p2auc:  %.4f'%(test_p2auc))
    return test_p2auc

if __name__ == "__main__":
    
    json_path = sys.argv[2]
    if sys.argv[1].startswith('cifar'):
        json_path += '_cifar'
        path = sys.argv[1].strip().split("-")[1]
        # print(path)
        if path == '100':
            json_path += "_100"
    elif sys.argv[1] == 'tiny-imagenet-200':
        json_path += '_tiny_imagenet'
    
    datasets = {
        "data_dir": osp.join('./data', sys.argv[1]),
        "lmdb_dir": osp.join('./data/lmdb', sys.argv[1])
    }


    with open(osp.join("params", json_path + '.json'),'r') as f:
        print("load json from: ", osp.join("params", json_path + '.json'))
        args = json.load(f)
        args = edict(args)

    args.dataset.update(datasets)
    loss_type = sys.argv[2]

    args.training.save_path = osp.join('./save',
                                       sys.argv[1])

    if not osp.exists(args.training.save_path):
        os.makedirs(args.training.save_path)

    if loss_type not in ["SquareAUCLoss", 'TPAUCLoss']:
        raise ValueError("{} is not included".format(loss_type))

    test_p2auc = train(args, use_test=True, idx=0)
