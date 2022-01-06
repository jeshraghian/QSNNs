import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from snntorch.spikevision import spikedata

def load_data(config):
        data_dir = config['data_dir']
        
        # Note: train set / test set are of different durations
        # we used num_steps=100 due to memory limits
        # you will likely to improve our reported results by increasing num_steps=100 to 150

        trainset = spikedata.DVSGesture(data_dir, train=True, num_steps=100, dt=3000, ds=4)
        testset = spikedata.DVSGesture(data_dir, train=False, num_steps=600, dt=3000, ds=4)

        return trainset, testset