import torch
from torch.utils.data import Sampler, DataLoader
import numpy as np
import _pickle as pk
from collections import Counter
from sklearn.utils import shuffle
import math
import pandas as pd

import pdb 


class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, 
                class_vector, 
                batch_size, 
                rpos=1, 
                rneg=4, 
                random_state=7):
        self.class_vector = class_vector
        # self.n_splits = int(class_vector.size(0) / batch_size)
        self.batch_size = batch_size

        self.rpos = rpos
        self.rneg = rneg

        if isinstance(class_vector, torch.Tensor):
            y = class_vector.cpu().numpy()
        else:
            y = np.array(class_vector)

        y_counter = Counter(y)
        self.data = pd.DataFrame({'y': y})

        # print(self.data)

        # self.class_batch_size = {
        #     k: math.ceil(n * batch_size / y.shape[0])
        #     for k, n in y_counter.items()
        # }

        # print(self.class_batch_size)

        # print("number of images: %d" % len(y))
        # print("number of pos images: %d" % y_counter[1])
        # print("number of pos images: %d" % y_counter[0])

        # print(len(y_counter.keys()))
        # only implemented for binary classification, 1:pos, 0:neg
        if len(y_counter.keys()) == 2:
            
            ratio = (rneg, rpos)
            
            self.class_batch_size = {
            k: math.ceil(batch_size * ratio[k] / sum(ratio))
            for k in y_counter.keys()
            }

            # print(self.class_batch_size)

            if rpos / rneg > y_counter[1] / y_counter[0]:
                add_pos = math.ceil(rpos / rneg * y_counter[0]) - y_counter[1]

                print("-" * 50)
                print("To balance ratio, add %d pos imgs (with replace = True)" % add_pos)
                # print(add_pos)
                print("-" * 50)

                pos_samples = self.data[self.data.y == 1].sample(add_pos, replace=True)

                assert pos_samples.shape[0] == add_pos
                # print(pos_samples.shape)

                self.data = self.data.append(pos_samples, ignore_index=False)

            else:
                add_neg = math.ceil(rneg / rpos * y_counter[1]) - y_counter[0]

                print("-" * 50)
                print("To balance ratio, add %d neg imgs repeatly" % add_neg)
                print("-" * 50)

                neg_samples = self.data[self.data.y == 0].sample(add_neg, replace=True)

                assert neg_samples.shape[0] == add_neg

                self.data = self.data.append(neg_samples, ignore_index=False)

        print("-" * 50)
        print("after complementary the ratio, having %d images" % self.data.shape[0])
        print("-" * 50)

        self.real_batch_size = int(sum(self.class_batch_size.values()))

    def gen_sample_array(self):
        # sampling for each class
        def sample_class(group):
            n = self.class_batch_size[group.name]
            return group.sample(n) if group.shape[0] >= n else group.sample(n, replace=True)


        # sampling for each batch
        data = self.data.copy()

        data['idx'] = data.index

        data = data.reset_index()

        # print("data: %d" % len(data))

        result = []
        while True:
            try:
                batch = data.groupby('y', group_keys=False).apply(sample_class)
                assert len(
                    batch) == self.real_batch_size, 'not enough instances!'
            except (ValueError, AssertionError) as e:
                # print(e)
                break
            # print('sampled a batch ...')
            result.extend(shuffle(batch.idx))
            # pdb.set_trace()

            data.drop(index=batch.index, inplace=True)
        return result

    def __iter__(self):
        self.index_list = self.gen_sample_array()
        return iter(self.index_list)

    def __len__(self):
        try:
            l = len(self.index_list)
        except:
            l = len(self.class_vector)
        return l
