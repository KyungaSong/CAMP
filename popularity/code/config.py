import os.path
import torch
from datetime import datetime


class Config(object):
    def __init__(self, args):
        self.alpha = args.alpha
        