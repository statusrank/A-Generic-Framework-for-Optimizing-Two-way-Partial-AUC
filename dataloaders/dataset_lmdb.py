import torch
import numpy as np
import os
import os.path as osp
import cv2
import pandas as pd
import time
import pickle
import lmdb
from tqdm import tqdm

from .base_dataset import BaseDataset
from .base_dataset import pil_loader


def build_lmdb(save_path, metas, commit_interval=1000):
    if not save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with 'lmdb'.")
    if osp.exists(save_path):
        print('Folder [{:s}] already exists.'.format(save_path))
        return

    if not osp.exists('/'.join(save_path.split('/')[:-1])):
        os.makedirs('/'.join(save_path.split('/')[:-1]))

    data_size_per_img = cv2.imread(metas[0][0], cv2.IMREAD_UNCHANGED).nbytes
    data_size = data_size_per_img * len(metas)
    env = lmdb.open(save_path, map_size=data_size * 10)
    txn = env.begin(write=True)

    shape = dict()

    print('Building lmdb...')
    for i in tqdm(range(len(metas))):
        image_filename = metas[i][0]
        img = pil_loader(filename=image_filename)
        assert img is not None and len(img.shape) == 3

        txn.put(image_filename.encode('ascii'), img.copy(order='C'))
        shape[image_filename] = '{:d}_{:d}_{:d}'.format(img.shape[0], img.shape[1], img.shape[2])

        if i % commit_interval == 0:
            txn.commit()
            txn = env.begin(write=True)
    
    pickle.dump(shape, open(os.path.join(save_path, 'meta_info.pkl'), "wb"))

    txn.commit()
    env.close()
    print('Finish writing lmdb.')


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

            if not osp.isdir(dir):
                return []

            file_list = os.listdir(dir)
            ret = []
            for i in file_list:
                ret += get_all_files(osp.join(dir, i), ext)
            return ret


        self.args = args
        self.tmp = None 
        self.data = None 
        self.targets = None

        if args.get('npy_style', False):
            self.tmp = np.load(self.data_dir + '.npy', allow_pickle=True).item()
            self.data = self.tmp['data']
            self.targets = self.tmp['targets']
            assert len(self.data) == len(self.targets)
            self.img_list = ['%08d'%i for i in range(len(self.data))]
            self.metas = []
            for i in range(len(self.targets)):
                cls_id = self.class2id.get(str(self.targets[i]), 0)
                if cls_id >= 0:
                    self.metas.append((self.data[i], cls_id))
            args.use_lmdb = False
            self.args.use_lmdb = False
        else:
            self.img_list = get_all_files(self.data_dir, ['jpg', 'jpeg', 'png'])
            self.metas = []
            for i in self.img_list:
                cls_id = self.class2id.get(i.split('/')[-2], 0)
                if cls_id >= 0:
                    self.metas.append((i, cls_id))
            # self.metas = [(i, self.class2id[i.split('/')[-2]]) for i in self.img_list]

        self._num = len(self.metas)
        print('%s set has %d images' % (self.split, self.__len__()))

        if args.get('use_lmdb', False):
            self.lmdb_dir = osp.join(args.lmdb_dir, split + '.lmdb')
            build_lmdb(self.lmdb_dir, self.metas)
            self.initialized = False
            self._load_image = self._load_image_lmdb
        else:
            self._load_image = self._load_image_pil


        self._labels = [i[1] for i in self.metas]
        self._cls_num_list = pd.Series(self._labels).value_counts().sort_index().values
        self._freq_info = [
            num * 1.0 / sum(self._cls_num_list) for num in self._cls_num_list
        ]
        self._num_classes = len(self._cls_num_list)
        self._class_dim = len(set(self._labels))
        print('class number: ', self._cls_num_list)

    def _init_lmdb(self):
        if not self.initialized:
            env = lmdb.open(self.lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False)
            self.lmdb_txn = env.begin(write=False)
            self.meta_info = pickle.load(open(os.path.join(self.lmdb_dir, 'meta_info.pkl'), "rb"))
            self.initialized = True

    def _load_image_lmdb(self, img_filename):
        self._init_lmdb()
        img_buff = self.lmdb_txn.get(img_filename.encode('ascii'))
        C, H, W = [int(i) for i in self.meta_info[img_filename].split('_')]
        img = np.frombuffer(img_buff, dtype=np.uint8).reshape(C, H, W)
        return img

    def _load_image_pil(self, img_filename):
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
            'image': self._load_image(self.metas[idx][0]),
            'label': self.metas[idx][1]
        }
        sample = self.transform(sample)
        return sample['image'], sample['label']
    
    def _getitem_npy(self, idx):
        sample = {
            'image': self.metas[idx][0],
            'label': self.metas[idx][1]
        }
        sample = self.transform(sample)
        return sample['image'], sample['label']

    def __getitem__(self, idx):
        if self.args.get('npy_style', False):
            return self._getitem_npy(idx)
        return self._getitem(idx)
