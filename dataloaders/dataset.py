import torch
import numpy as np
import os
import os.path as osp
import cv2
import pandas as pd
import time

from .base_dataset import BaseDataset
from .base_dataset import pil_loader


class Dataset(BaseDataset):
    def __init__(self, args, split='train', **kwargs):
        super().__init__(args)
        self.data_dir = osp.join(args.data_dir, split)
        self.class2id = args.get('class2id', {'pos': 1, 'neg': 0})
        self.split = split
        if split == 'train':
            self.transform = self.transform_train()
        elif split == 'val':
            self.transform = self.transform_validation()
        elif split == 'test':
            self.transform = self.transform_validation()
        else:
            raise ValueError

        def get_all_files(dir, ext):
            for e in ext:
                if dir.lower().endswith(e):
                    return [dir]

            file_list = os.listdir(dir)
            ret = []
            for i in file_list:
                ret += get_all_files(osp.join(dir, i), ext)
            return ret

        self.img_list = get_all_files(self.data_dir, ['jpg', 'jpeg', 'png'])
        self.metas = [(i, self.class2id[i.split('/')[-2]]) for i in self.img_list]

        self._num = len(self.metas)
        print('%s set has %d images' % (self.split, self.__len__()))
        # logger.info('%s set has %d images' % (self.split, self.__len__()))

        self._labels = [i[1] for i in self.metas]
        self._cls_num_list = pd.Series(self._labels).value_counts().sort_index().values
        self._freq_info = [
            num * 1.0 / sum(self._cls_num_list) for num in self._cls_num_list
        ]
        self._num_classes = len(self._cls_num_list)
        self._class_dim = len(set(self._labels))

    def load_image(self, img_filename):
        return pil_loader(img_filename)

    def get_class_dim(self):
        return self._class_dim

    def get_labels(self):
        return self._labels

    def get_cls_num_list(self):
        return self._cls_num_list

    def get_freq_info(self):
        return self._freq_info

    def get_num_classes(self):
        return self._num_classes

    def __len__(self):
        return self._num

    def __str__(self):
        return self.args.data_dir + '  split=' + str(self.split)

    def _getitem(self, idx):
        sample = {
            'image': self.load_image(self.metas[idx][0]),
            'label': self.metas[idx][1]
        }
        sample = self.transform(sample)
        return sample['image'], sample['label']

    def __getitem__(self, idx):
        return self._getitem(idx)
