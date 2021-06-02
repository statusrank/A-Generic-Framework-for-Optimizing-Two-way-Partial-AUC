'''
copyright: Shilong Bao 
email: baoshilong@iie.ac.cn
'''

import getpass
import logging
import sys
import torch
import numpy as np
# import torch
import random
import os 
class MyLog(object):
    def __init__(self, init_file=None,name = None):
        user = getpass.getuser()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        if init_file == None:
            assert False
            # logFile = sys.argv[0][0:-3] + '.log'
        else:
            logFile = init_file
        # formatter = logging.Formatter('%(asctime)-12s %(levelname)-8s %(name)-10s %(message)-12s')
        formatter = logging.Formatter('')
        logHand = logging.FileHandler(logFile, encoding="utf8")
        logHand.setFormatter(formatter)
        logHand.setLevel(logging.INFO)

        logHandSt = logging.StreamHandler()
        logHandSt.setFormatter(formatter)

        self.logger.addHandler(logHand)
        self.logger.addHandler(logHandSt)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warn(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
