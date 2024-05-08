import os.path
import torch
from datetime import datetime


class Config(object):
    def __init__(self, args):
        self.alpha = args.alpha
        self.batch_size = args.batch_size 
        self.lr = args.lr 
        self.time_unit = args.time_unit 
        self.pop_time_unit = args.pop_time_unit 
        self.data_preprocessed = args.data_preprocessed
        self.num_epochs = args.num_epochs
        self.embedding_dim = args.embedding_dim

        