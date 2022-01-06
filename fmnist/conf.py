import torch

# For high precision training, set 'num_bits' to 'None'
# For N-bit quantization-aware training, where N is of type `int`, set 'num_bits' to N

config = {

        # experiment setup
        'exp_name' : 'snn_fmnist',
        'num_trials' : 3,
        'num_epochs' : 100,
        'data_dir' : "~/data/fmnist",

        # data recording
        'save_csv' : True,
        'save_model' : True,
        'plot_loss' : True, # to plot multiple trials, see plot_loss.py

        # neuron hyperparams 
        'grad_clip' : True,
        'weight_clip' : True,
        'batch_norm' : True,
        'dropout' : 0.13,
        'beta' : 0.39,
        'threshold' : 1.47,
        'lr' : 2e-3,
        'slope': 7.66,

        # network hyperparams
        'num_bits' : None, # none for flt32; N (type: int) for n-bit quantization
        'num_steps' : 100,

        # batching
        'batch_size' : 128,
        'seed' : 0,
        'num_workers' : 0,
        
        # optimization hyperparams
        'correct_rate': 0.8,
        'incorrect_rate' : 0.2,
        'betas' : (0.9, 0.999),
        'early_stopping': False,
        'patience': 100,

        # cosine annealing schedule
        't_max' : 4690,  # period of cosine scheduler [num iterations] -- i.e., cycle every 10 epochs
        'eta_min' : 0,

    }

def optim_func(net, config):
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"], betas=config['betas'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['t_max'], eta_min=config['eta_min'], last_epoch=-1)
    return optimizer, scheduler