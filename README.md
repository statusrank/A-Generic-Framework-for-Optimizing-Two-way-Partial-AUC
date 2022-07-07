# When All We Need is a Piece of the Pie: A Generic Framework for Optimizing Two-way Partial AUC
>  Zhiyong Yang, Qianqian Xu, Shilong Bao, Yuan He, Xiaochun Cao and Qingming Huang. [When All We Need is a Piece of the Pie: A Generic Framework for Optimizing Two-way Partial AUC](https://github.com/statusrank/A-Generic-Framework-for-Optimizing-Two-way-Partial-AUC/blob/main/TPAUC.pdf). ICML 2021 (Long talk, 3\%)

This is an official implementation with PyTorch, and we run our code on Ubuntu 18.04 server. More experimental details can be found in our paper.

# To all interested researchers
**The authors have open-sourced an end-to-end PyTorch library for X metrics learning (including partial AUC studied by this work and other state-of-the-art x-metric-learning algorithms) in machine learning**. Please refer to [XCurve](https://github.com/statusrank/XCurve) for more details. We hope our library can help you deploy/attain your ML model conveniently and easily. Thanks for all!  

# Update
- 2022.7.7 ----- The authors notice that the reproduction of our proposed algorithms on small datasets, such as CIFAR-10 and CIFAR-100, seems not to be stable, since the version/environment upgraded, tiny initialization changed, or some other potential reasons. But we also observe that the results on the larger dataset such as TINY-IMAGENET-200 are the same or even better than those reported in our paper. Due to the limitations, we refer the readers to our [XCurve](https://github.com/statusrank/XCurve/tree/master/example) library to see how to run our algorithms in the TINY-IMAGENET-200 dataset. Thanks for all!



# Dependencies
- python 3.7+
- pytorch 1.8+
- numpy
- tqdm
- scikit-learn
- scikit-image
- lmdb
- easydict

# Abstract

![TPAUC](https://github.com/statusrank/A-Generic-Framework-for-Optimizing-Two-way-Partial-AUC/blob/main/img/TPAUC.png)

we present the first trial in this paper to optimize this new metric. The critical challenge along this course lies in the difficulty of performing gradient-based optimization with end-to-end stochastic training, even with a proper choice of surrogate loss. To address this issue, we propose a generic framework to construct surrogate optimization problems, which supports efficient end-to-end training with deep-learning. Moreover, our theoretical analyses show that: 1) the objective function of the surrogate problems will achieve an upper bound of the original problem under mild conditions, and 2) optimizing the surrogate problems leads to good generalization performance in terms of TPAUC with a high probability. Finally, empirical studies over several benchmark datasets speak to the efficacy of our

# Experiments

![Exp](https://github.com/statusrank/A-Generic-Framework-for-Optimizing-Two-way-Partial-AUC/blob/main/img/Exp.png)


# How to Run
- First of all, u need to install the dependencies by 
```
pip3 install -r requirements.txt
```
- Then you need to set the corresponding params on the params/*.json file. 
For example, set the corresponding class-id as 1 in "class2id" to run on the different subsets.

- Demo of training
```
  CUDA_VISIBLE_DEVICES=3 python3 train.py dataset-name method-name
```
For instance, 
```
CUDA_VISIBLE_DEVICES=0 python3 train.py cifar-10-long-tail TPAUCLoss
```

You can also config run.sh and run it:

```
chmod +x run.sh
./run.sh
```
## Additional Explanation
In terms of the hyper-parameters details, it can be found in the paper's [supp material](https://github.com/statusrank/A-Generic-Framework-for-Optimizing-Two-way-Partial-AUC/blob/main/TPAUC.pdf).

Note that, for the hyper-parameter \gamma(r) of Poly model, there is a litter difference. The best parameter gamma'(in our implementation) = 1 / (r - 1). 

# Citation
Please cite our paper if you use this code in your own work.

```
@inproceedings{DBLP:conf/icml/YQBYXQ, 
author    = {Zhiyong Yang, Qianqian Xu, Shilong Bao, Yuan He, Xiaochun Cao and Qingming Huang},
  title     = {When All We Need is a Piece of the Pie: A Generic Framework for Optimizing Two-way Partial AUC},
  booktitle = {ICML},
  pages     = {11820--11829},
  year      = {2021}

```

# Contact

If there are any questions, contact baoshilong@iie.ac.cn or yangzhiyong21@ucas.ac.cn
