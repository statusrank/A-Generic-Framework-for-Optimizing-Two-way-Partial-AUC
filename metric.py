import abc
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import math
def argTopk(score, k):
    return argBottomk(-score, k)

def argBottomk(score, k):
    
    return devec(np.argpartition(devec(score), k)[:k])

def vec(x):
    return x.reshape(-1,1)

def devec(x):
    return x.squeeze() if x.shape[0] > 1 else x 

def subIndexNegPosK(score, label, rpos, rneg):
    negIndex = np.where(label != 1)[0]
    posIndex = np.where(label == 1)[0]
    nP, nN = len(posIndex), len(negIndex)
    kpos = math.floor(nP * rpos)
    kneg = math.floor(nN * rneg)
    negTopIndex = negIndex[argTopk(score[negIndex], kneg)]
    posBotIndex = posIndex[argBottomk(score[posIndex], kpos)]

    return devec(np.vstack((vec(posBotIndex), vec(negTopIndex))))

def p2AUC(y_true, y_pred, alpha, beta):
    subIndex = subIndexNegPosK(y_pred, y_true, alpha, beta)
    y_true = y_true[subIndex]
    y_pred = y_pred[subIndex]
    return roc_auc_score(y_true, y_pred)