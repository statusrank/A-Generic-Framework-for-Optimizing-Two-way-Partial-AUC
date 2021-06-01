# When All We Need is a Piece of the Pie: A Generic Framework for Optimizing Two-way Partial AUC
>  Zhiyong Yang, Qianqian Xu, Shilong Bao, Yuan He, Xiaochun Cao and Qingming Huang. [When All We Need is a Piece of the Pie: A Generic Framework for Optimizing Two-way Partial AUC](https://github.com/statusrank/A-Generic-Framework-for-Optimizing-Two-way-Partial-AUC/blob/main/TPAUC.pdf). ICML 2021 (Long talk, 3\%)

This is an official implementation with PyTorch, and we run our code on Ubuntu 18.04 server. More experimental details can be found in our paper.


# Abstract

![TPAUC](https://github.com/statusrank/A-Generic-Framework-for-Optimizing-Two-way-Partial-AUC/blob/main/img/TPAUC.png)

we present the first trial in this paper to optimize this new metric. The critical challenge along this course lies in the difficulty of performing gradient-based optimization with end-to-end stochastic training, even with a proper choice of surrogate loss. To address this issue, we propose a generic framework to construct surrogate optimization problems, which supports efficient end-to-end training with deep-learning. Moreover, our theoretical analyses show that: 1) the objective function of the surrogate problems will achieve an upper bound of the original problem under mild conditions, and 2) optimizing the surrogate problems leads to good generalization performance in terms of TPAUC with a high probability. Finally, empirical studies over several benchmark datasets speak to the efficacy of our

# Experiments

![Exp](https://github.com/statusrank/A-Generic-Framework-for-Optimizing-Two-way-Partial-AUC/blob/main/img/Exp.png)

# Citation
Please cite our paper if you use this code in your own work.

```
@inproceedings{DBLP:conf/icml/YQBYXQ, 
author    = {Zhiyong Yang, Qianqian Xu, Shilong Bao, Yuan He, Xiaochun Cao and Qingming Huang},
  title     = {When All We Need is a Piece of the Pie: A Generic Framework for Optimizing Two-way Partial AUC},
  booktitle = {ICML},
  pages     = {},
  year      = {2021}

```

