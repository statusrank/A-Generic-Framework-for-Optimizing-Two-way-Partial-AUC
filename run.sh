# !/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 train.py cifar-10-long-tail SquareAUCLoss
CUDA_VISIBLE_DEVICES=0 python3 train.py cifar-10-long-tail TPSquareAUCLoss