from torch.utils.data import DataLoader
from .dataset_lmdb import Dataset
from .sampler import StratifiedSampler


def get_datasets(args):
    train_set = Dataset(args, 'train')
    val_set = Dataset(args, 'test')
    test_set = Dataset(args, 'val')
    return train_set, val_set, test_set

def get_data_loaders(train_set,
                     val_set,
                     test_set,
                     train_batch_size,
                     test_batch_size,
                     num_workers=4,
                     rpos = 1,
                     rneg = 4,
                     random_state = 1234):
    sampler = StratifiedSampler(train_set.get_labels(),
                                train_batch_size,
                                rpos = rpos,
                                rneg = rneg,
                                random_state=random_state)
                                
    train_loader = DataLoader(train_set,
                              batch_size=sampler.real_batch_size,
                            #   shuffle=True,
                              sampler=sampler,
                              num_workers=num_workers)
    val_loader = DataLoader(val_set,
                            batch_size=test_batch_size,
                            shuffle=True,
                            num_workers=num_workers)
    test_loader = DataLoader(test_set,
                             batch_size=test_batch_size,
                             shuffle=True,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader


__all__ = ['Dataset', 'get_datasets', 'get_data_loaders', 'StratifiedSampler']
