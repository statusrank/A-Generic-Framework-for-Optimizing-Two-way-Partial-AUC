import abc
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from cvxopt import matrix, spmatrix
from cvxopt.solvers import options, qp
import numpy as np
import os
import math
from sklearn.svm import SVC

def argTopk(score, k):
    return argBottomk(-score, k)

def argBottomk(score, k):
    
    return devec(np.argpartition(devec(score), k)[:k])

def vec(x):
    return x.reshape(-1,1)

def devec(x):
    return x.squeeze() if x.shape[0] > 1 else x 


def subIndexNegK(score, label, rneg):
    negIndex = np.where(label != 1)[0]
    posIndex = np.where(label == 1)[0]
    nN = len(negIndex)
    kneg = math.floor(nN * rneg)
    negTopIndex = negIndex[argTopk(score[negIndex], kneg)]
    return devec(np.vstack((vec(posIndex), vec(negTopIndex))))


def subIndexNegPosK(score, label, rpos, rneg):
    negIndex = np.where(label != 1)[0]
    posIndex = np.where(label == 1)[0]
    nP, nN = len(posIndex), len(negIndex)
    kpos = math.floor(nP * rpos)
    kneg = math.floor(nN * rneg)
    negTopIndex = negIndex[argTopk(score[negIndex], kneg)]
    posBotIndex = posIndex[argBottomk(score[posIndex], kpos)]

    return devec(np.vstack((vec(posBotIndex), vec(negTopIndex))))


def cvxopt_matrix(M):
    if type(M) is np.ndarray:
        return matrix(M)
    elif type(M) is spmatrix or type(M) is matrix:
        return M
    coo = M.tocoo()
    return spmatrix(
        coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=M.shape)


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None, solver=None,
                    initvals=None):
    """
    Solve a Quadratic Program defined as:

        minimize
            (1/2) * x.T * P * x + q.T * x

        subject to
            G * x <= h
            A * x == b

    using CVXOPT <http://cvxopt.org/>.

    Parameters
    ----------
    P : numpy.array, cvxopt.matrix or cvxopt.spmatrix
        Symmetric quadratic-cost matrix.
    q : numpy.array, cvxopt.matrix or cvxopt.spmatrix
        Quadratic-cost vector.
    G : numpy.array, cvxopt.matrix or cvxopt.spmatrix
        Linear inequality matrix.
    h : numpy.array, cvxopt.matrix or cvxopt.spmatrix
        Linear inequality vector.
    A : numpy.array, cvxopt.matrix or cvxopt.spmatrix
        Linear equality constraint matrix.
    b : numpy.array, cvxopt.matrix or cvxopt.spmatrix
        Linear equality constraint vector.
    solver : string, optional
        Set to 'mosek' to run MOSEK rather than CVXOPT.
    initvals : numpy.array, optional
        Warm-start guess vector.

    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.

    Note
    ----
    CVXOPT only considers the lower entries of `P`, therefore it will use a
    wrong cost function if a non-symmetric matrix is provided.
    """
    args = [cvxopt_matrix(P), cvxopt_matrix(q)]
    if G is not None:
        args.extend([cvxopt_matrix(G), cvxopt_matrix(h)])
        if A is not None:
            args.extend([cvxopt_matrix(A), cvxopt_matrix(b)])
    sol = qp(*args, solver=solver, initvals=initvals)
    # if 'optimal' not in sol['status']:
    #     return None
    return np.array(sol['x']).reshape((q.shape[0],))

def auc_binary(y_true, y_pred):
    if len(y_true.shape) > 1:
        y_true = y_true.squeeze()

    if len(y_pred.shape) > 1:
        y_pred = y_pred.squeeze()

    label = y_true == 1
    nP = label.sum()
    nN = label.shape[0] - nP
    sindex = np.argsort(y_pred)
    lSorted = label[sindex]
    auc = (np.where(lSorted != True) - np.arange(nN)).sum()
    auc /= (nN * nP)

    return 1 - auc

def paucZero(y_true, y_pred, beta):
    subIndex = subIndexNegK(y_pred, y_true, beta)
    y_true = y_true[subIndex]
    y_pred = y_pred[subIndex]
    return auc_binary(y_true, y_pred)

def p2AUC(y_true, y_pred, alpha, beta):
    subIndex = subIndexNegPosK(y_pred, y_true, alpha, beta)
    y_true = y_true[subIndex]
    y_pred = y_pred[subIndex]
    return roc_auc_score(y_true, y_pred)
    #return auc_binary(y_true, y_pred)

def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


class StructAUCAbs(BaseEstimator):
    def __init__(self, epsilon=1e-10, constant=100, max_iter=50):
        self.w = None
        self.Q = None
        self.S = None
        self.loss = None
        self.constant = constant
        self.epsilon = epsilon
        self.C = []
        self.X = None
        self.y = None
        self.joint_feature = None
        self.c = None
        self.nP = None
        self.nN = None
        self.s = None
        self.fac = None
        self.max_iter = max_iter
        self.Xsub = None
        self.ysub = None
        self.facP = None 
        self.facN = None
        self.score = None
        self.score_sub = None
    # @abc.abstractmethod
    # def _findMostViolated(self):
    #     # self.X, self.y
    #     pass

    @abc.abstractmethod
    def _calFactor(self):
        # self.X, self.y
        pass

    def _calC(self, s):
        # s = np.zeros(self.y.shape[0])
        c = np.zeros(self.ysub.shape[0])
        # self.s = np.matmul(self.X, self.w).squeeze()
        sortedIndex = np.argsort(s)[::-1]
        labelSorted = self.ysub[sortedIndex].copy().squeeze()

        posIndex = np.where(labelSorted == 1)[0]
        negIndex = np.where(labelSorted != 1)[0]
        cN = negIndex - np.arange(self.facN)
        cP = posIndex - np.arange(self.facP)
        cP = self.facN - cP
        c[sortedIndex[posIndex]] = cP
        c[sortedIndex[negIndex]] = -cN

        return c.reshape([-1, 1])

    @abc.abstractmethod
    def _calLoss(self, y_hat):
        pass

    def _calJointFeature(self, yhat):

        return np.matmul(self.Xsub.T, yhat).squeeze() / self.fac


    def _calJointFeatureInit(self):
        posIndex = devec(self.ysub == 1)
        negIndex = devec(self.ysub != 1)
        ap = self.facN * np.ones((self.facP, 1))
        aN = - self.facP * np.ones((self.facN, 1))
        a = np.zeros((self.ysub.shape[0], 1))
        a[posIndex] = ap
        a[negIndex] = aN
        self.joint_feature = np.matmul(self.Xsub.T, a).squeeze() / self.fac
    
    
    @abc.abstractmethod
    def _updateSubSample(self):
        pass
    
    def predict(self, X):
        return np.matmul(X, self.w)

    def _findMostViolated(self):

        negIndex = (self.ysub != 1).squeeze()
        scorep = self.score_sub.copy()
        scorep[negIndex] = 1 + scorep[negIndex]
        yhatInstanceWise = self._calC(scorep)
        new_feat = self._calDeltaPsi(yhatInstanceWise)
        new_loss = self._calLoss(yhatInstanceWise)
        precision = new_loss - np.matmul(new_feat, self.w)

        return yhatInstanceWise, precision, new_feat, new_loss
    # def score(self, X, y):
    #     return np.sum(self.predict(X) != y)/y.shape[0]

    def _calInfo(self):
        self.nP = (self.y == 1).sum()
        self.nN = (self.y != 1).sum()

    def fit(self, X, y):

        self.X = X

        if len(y.shape) == 1:
            y = y[:, np.newaxis]

        self.y = y
        self._calInfo()
        self.w = np.zeros((self.X.shape[1], 1))
        # self.w = np.random.rand(self.X.shape[1], 1)
  
        self._calFactor()
        # line 3
        for iter_time in range(self.max_iter):
            # line 4-5
            self.iter =iter_time
            self.score = devec(np.matmul(self.X, self.w))
            ## update X_sub y_sub score_sub
            self._updateSubSample()
            ### update joint_feature 
            self._calJointFeatureInit()
            
            print('iteration times:{:3d}'.format(iter_time))

            yhat, precision, new_feat, new_loss = self._findMostViolated()

            if iter_time > 0:
                ksi = self._calKsi()
            else:
                ksi = 0

            # update by the new sample


            if precision > ksi + self.epsilon:
                # line 7
                # line 8
                self._updateS(new_feat)
            # line 6
                self._updateLoss(new_loss)
                self._updateQ(new_feat)

                if iter_time == 0:
                    alpha = - self.loss / self.Q
                    alpha = min(alpha, self.constant)
                else:
                    G = np.ones((1, iter_time+1))
                    G = np.vstack((G, -np.eye(iter_time + 1)))
                    h = np.array([self.constant])
                    lb = np.zeros(iter_time+1)
                    h = np.hstack((h, lb))
                    alpha = cvxopt_solve_qp(self.Q, self.loss, G, h)
                    if len(alpha.shape) == 1:
                        alpha = alpha.reshape([-1, 1])
                if iter_time == 0:
                    self.w = alpha * self.S[:, np.newaxis]
                else:
                    self.w = np.matmul(self.S.T, alpha)
            else:
                if iter_time > 0:
                    break

        return self

    def _calDeltaPsi(self, y_hat):
        return self.joint_feature - self._calJointFeature(y_hat)

    def _updateS(self, new_feat):
        if self.S is None:
            if len(new_feat.shape) == 1:
                new_feat = new_feat.squeeze()
            self.S = new_feat
        else:
            if len(new_feat.shape) == 1:
                new_feat = new_feat.squeeze()

            self.S = np.vstack((self.S, new_feat))

    def _calKsi(self):

        lossVec = (-np.matmul(self.S, self.w) -self.loss.reshape([-1, 1])).max()

        return np.max(lossVec, 0)

    def _updateQ(self, new_feat):
        if self.Q is None:
            self.Q = np.matmul(self.S, self.S.T).reshape(1,)
        else:
            delta_Psi_y_new = new_feat
            c = np.linalg.norm(delta_Psi_y_new)**2
            self.q = np.dot(self.S[:-1, :], delta_Psi_y_new)
            if self.Q.shape[0] == 1:
                self.Q = np.hstack(
                    (self.Q.reshape([-1, 1]), self.q.reshape([-1, 1])))
                self.Q = np.vstack(
                    (self.Q, np.hstack(
                        (self.q.reshape([1, -1]), np.array(c).reshape([1, -1]))))
                )
            else:
                self.Q = np.hstack((self.Q, self.q.reshape([-1, 1])))
                self.Q = np.vstack(
                    (self.Q, np.hstack((self.q.reshape([1, -1]), np.array(c).reshape([1, -1])))))

    def _updateLoss(self, new_loss):
        if self.loss is None:
            self.loss = np.array([-new_loss])
        else:
            self.loss = np.hstack((self.loss, -new_loss)).reshape((-1, ))


class StructAUCAll(StructAUCAbs):
    def __init__(self, epsilon=0.2, C=1, max_iter=100):
        super().__init__(epsilon, C, max_iter)

    def _calFactor(self):
        self.fac = self.nP * self.nN
        self.facP = self.nP
        self.facN = self.nN

    def _updateSubSample(self):

        if self.iter == 0:
            self.Xsub  = self.X.copy()
            self.ysub  = self.y.copy()
            self.score_sub = self.score.copy()

    def _calJointFeatureInit(self):
        if self.iter ==0:
            posIndex = devec(self.ysub == 1)
            negIndex = devec(self.ysub != 1)
            ap = self.facN * np.ones((self.facP, 1))
            aN = -self.facP * np.ones((self.facN, 1))
            a = np.zeros((self.ysub.shape[0], 1))
            a[posIndex] = ap
            a[negIndex] = aN
            self.joint_feature = np.matmul(self.X.T, a).squeeze() / self.fac


    def _calLoss(self, yhat):
        return 0.5 * (self.facP + yhat[self.ysub != 1]).sum() / self.fac


class StructPAUCZero(StructAUCAbs):
    def __init__(self, epsilon=0.2, C=1, max_iter=100, rneg = 0.1):
        super().__init__(epsilon, C, max_iter)
        self.rneg = rneg

    def _calFactor(self):
        self.facP = self.nP
        self.facN = math.floor(self.nN * self.rneg)
        self.fac = self.facP * self.facN


    def _updateSubSample(self):
        subIndex = subIndexNegK(self.score, self.y, self.rneg)
        self.Xsub = self.X[subIndex]
        self.ysub = self.y[subIndex]
        self.score_sub = self.score[subIndex]

    def _calLoss(self, yhat):
        return 0.5 * (self.facP + yhat[self.ysub != 1]).sum() / self.fac

class StructP2AUC(StructAUCAbs):
    def __init__(self,epsilon=0.2, C=1, max_iter=100, rpos = 0.1, rneg= 0.1):
        super().__init__(epsilon, C, max_iter)
        self.rpos = rpos
        self.rneg = rneg

    def _calFactor(self):
        self.facP = math.floor(self.nP * self.rpos)
        self.facN = math.floor(self.nN * self.rneg)
        self.fac = self.facP * self.facN


    def _updateSubSample(self):
        subIndex = subIndexNegPosK(self.score, self.y, self.rpos, self.rneg)
        self.Xsub = self.X[subIndex]
        self.ysub = self.y[subIndex]
        self.score_sub = self.score[subIndex]


    def _calLoss(self, yhat):
        return 0.5 * (self.facP + yhat[self.ysub != 1]).sum() / self.fac


if __name__ == '__main__':
    svmperf = StructAUCAll(C=0.0005, epsilon=1e-10, max_iter = 1000)

    clf3 = SVC(gamma=.1, probability=True,max_iter=50)
    clf1 = DecisionTreeClassifier(max_depth=50)
    # n_samples = 40000
    # n_features = 400
    # X = 0.1 * np.random.randn(n_samples, 400)
    # w = 100 * np.random.randn(400, 1)
    # y = 1.0 * (np.matmul(X, w) + 10 * np.random.randn(n_samples, 1) > 0)
    # svmperf.fit(X, y)
    # clf1.fit(X, y)
    
    # print(svmperf.w)
    # print(auc_binary( y, svmperf.predict(X)))
    # print(auc_binary( y, clf1.predict(X)))
    # load data
    data_path = 'breastcancer'

    trainX = np.load(os.path.join(data_path, 'train_X.npy'))
    trainY = np.load(os.path.join(data_path, 'train_Y.npy'))

    testX = np.load(os.path.join(data_path, 'test_X.npy'))
    testY = np.load(os.path.join(data_path, 'test_Y.npy'))

    svmperf.fit(trainX, trainY)
    clf3.fit(trainX, trainY)
    clf1.fit(trainX, trainY)
    # print(auc_binary(trainY, svmperf.predict(trainX)))
    # print(auc_binary(testY, svmperf.predict(testX)))
    # # print('score: ' + str(svmperf.score(X, y)))
    # print(auc_binary(testY, clf3.predict(testX)))
    # print(auc_binary(testY, clf1.predict(testX)))

    # print(p2AUC(trainY, svmperf.predict(trainX),0.1,0.1))
    # print(p2AUC(testY, svmperf.predict(testX), 0.1, 0.1))
    # # print('score: ' + str(svmperf.score(X, y)))
    # print(p2AUC(testY, clf3.predict(testX),0.1, 0.1))
    # print(p2AUC(testY, clf1.predict(testX),0.1,0.1))


    print(roc_auc_score(trainY, svmperf.predict(trainX)))
    print(roc_auc_score(testY, svmperf.predict(testX)))
    # print('score: ' + str(svmperf.score(X, y)))
    print(roc_auc_score(testY, clf3.predict_proba(testX)[:,1]))
    print(roc_auc_score(testY, clf3.predict_proba(testX)[:,1]))
