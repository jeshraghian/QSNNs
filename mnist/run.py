import torch
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Net import Net
from evaluate import evaluate
from dataloader import load_data
from train import train
from test import test

config = {
    "exp_name": "MNIST",  # Experiment name
    "num_trials_eval": 3,  # Number of trails to execute (separate training and evaluation instances)
    "num_epochs_eval": 500,  # Number of epochs to train for (per trial)
    "data_dir": "~/data/",  # Data directory to download and store data
    "batch_size": 128,  # Batch size
    "seed": 0,  # Random seed
    "num_workers": 0,  # Number of workers for the dataloader
    "num_bits": 4,  # Bit resolution. If None, floating point resolution is used
    "save_csv": True,  # Whether or not to save loss, lr, and accuracy dataframes
    "early_stopping": True,  # Whether or not to use early stopping
    "patience": 100,  # Number of epochs to wait for improvement before stopping
    # Network parameters
    "grad_clip": False,  # Whether or not to clip gradients
    "weight_clip": False,  # Whether or not to clip weights
    "batch_norm": True,  # Whether or not to use batch normalization
    "dropout": 0.003168,  # Dropout rate
    "beta": 0.994,  # Decay rate parameter (beta)
    "threshold": 2.915,  # Threshold parameter (theta)
    "lr": 5.4e-3,  # Initial learning rate
    "slope": 13.84,  # Slope value (k)
    # Fixed params
    "num_steps": 100,  # Number of timesteps to encode input for
    "correct_rate": 0.8,  # Correct rate
    "incorrect_rate": 0.2,  # Incorrect rate
    "betas": (0.9, 0.999),  # Adam optimizer beta values
    "t_max": 4690,  # Frequency of the cosine annealing scheduler (5 epochs)
    "t_0": 4690,  # Initial frequency of the cosine annealing scheduler
    "t_mult": 2,  # The frequency of cosine is halved after every 4690 iters (10 epochs)
    "eta_min": 0,  # Minimum learning rate
}


def optim_func(net, config):
    optimizer = torch.optim.Adam(
        net.parameters(), lr=config["lr"], betas=config["betas"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["t_0"], eta_min=config["eta_min"], last_epoch=-1
    )
    loss_dependent = False
    return optimizer, scheduler, loss_dependent


if __name__ == "__main__":
    evaluate(Net, config, load_data, train, test, optim_func)
