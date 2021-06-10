# !/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 train.py cifar-10-long-tail Poly 
# CUDA_VISIBLE_DEVICES=0 python3 train.py cifar-10-long-tail Exp 

# CUDA_VISIBLE_DEVICES=0 python3 train.py cifar-10-long-tail CELoss 
# CUDA_VISIBLE_DEVICES=0 python3 train.py cifar-10-long-tail CBCE 
# CUDA_VISIBLE_DEVICES=0 python3 train.py cifar-10-long-tail CBFOCAL
# CUDA_VISIBLE_DEVICES=0 python3 train.py cifar-10-long-tail FOCAL 
CUDA_VISIBLE_DEVICES=0 python3 train.py cifar-10-long-tail SquareAUCLoss

# CUDA_VISIBLE_DEVICES=0 python3 train.py cifar-10-long-tail TPSquareAUCLoss