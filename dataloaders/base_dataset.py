import numpy as np
from torch.utils.data import Dataset
import os
from torchvision import transforms
from dataloaders import custom_transforms as tr
from abc import ABC, abstractmethod
import cv2
from PIL import Image, ImageFile
import scipy.io as scio


ImageFile.LOAD_TRUNCATED_IMAGES = True
def pil_loader(filename, label=False):
    ext = (os.path.splitext(filename)[-1]).lower()
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        img = Image.open(filename)
        if not label:
            img = img.convert('RGB')
            img = np.array(img).astype(dtype=np.uint8)
            img = img[:,:,::-1]  #convert to BGR
        else:
            if img.mode != 'L' and img.mode != 'P':
                img = img.convert('L')
            img = np.array(img).astype(dtype=np.uint8)
    elif ext == '.mat':
        img = scio.loadmat(filename)
    elif ext == '.npy':
        img = np.load(filename, allow_pickle=True)
    else:
        raise NotImplementedError('Unsupported file type %s'%ext)

    return img


class BaseDataset(Dataset,ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ignore_index = 255

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __str__(self):
        pass
    
    @staticmethod
    def modify_commandline_options(parser,istrain=False):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    def transform_train(self):
        temp = []
        temp.append(tr.Resize(self.args.input_size))

        if self.args.get('aug', True):
            print('\nWith augmentations.')
            temp.append(tr.RandomHorizontalFlip())
            temp.append(tr.RandomRotate(15))
            temp.append(tr.RandomCrop(self.args.input_size))
        else:
            print('\nWithout augmentations.')
        temp.append(tr.Normalize(self.args.norm_params.mean, self.args.norm_params.std))
        temp.append(tr.ToTensor())
        composed_transforms = transforms.Compose(temp)
        return composed_transforms

    def transform_validation(self):
        temp = []
        temp.append(tr.Resize(self.args.input_size))
        # temp.append(tr.RandomCrop(self.args.input_size))
        temp.append(tr.Normalize(self.args.norm_params.mean, self.args.norm_params.std))
        temp.append(tr.ToTensor())
        composed_transforms = transforms.Compose(temp)
        return composed_transforms
