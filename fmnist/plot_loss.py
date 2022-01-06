import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# local
from conf import *

def plot_func(config):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams["figure.figsize"] = (30, 10)
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['axes.linewidth'] = 2
    fig, ax = plt.subplots(1, 1)


    file_name = config['exp_name']
    loss_csv_list = []
    for trial in range(config['num_trials']):
        csv_name = 'loss_' + file_name + '_t' + str(trial) + '.csv'
        loss_csv_list.append(csv_name)

    data = {
        "FashionMNIST": {
            "cosine": loss_csv_list
        },
    }

    df = pd.DataFrame(columns=["dataset", "scheduler", "idx", "mean", "std"])
    for dataset_idx, dataset in enumerate(data.keys()):
        for scheduler in data[dataset].keys():
            grouped_trial_df = pd.DataFrame(columns=["idx", "loss"])
            for idx, trial in enumerate(data[dataset][scheduler]):
                trial_df = pd.read_csv(trial)
                trial_data = np.vstack((trial_df.index, trial_df.values.flatten())).T
                trial_df = pd.DataFrame(trial_data, columns=["idx", "loss"])
                grouped_trial_df = grouped_trial_df.append(trial_df)


            grouped_trial_df_ = grouped_trial_df.groupby('idx')
            grouped_trial_data = np.hstack((np.expand_dims(grouped_trial_df['idx'].unique(), 1), grouped_trial_df_.mean(), grouped_trial_df_.std()))
            grouped_trial_data = np.nan_to_num(grouped_trial_data, nan=0)
            trial_df = pd.DataFrame(grouped_trial_data, columns=['idx', 'mean', 'std'])
            trial_df["scheduler"] = scheduler
            trial_df["dataset"] = dataset
            df = df.append(trial_df)

    # store data to disk for very large files
    df.to_csv('loss_data.csv', index=False)

    df = pd.read_csv('loss_data.csv')
    
    # drop_count = 0
    # df = df[df['idx'] % drop_count == 0] # Downsample the plot resolution to reduce noise 

    ### Take moving averages to clean up
    df['mean_rolling'] = df.iloc[:,3].rolling(window=20).mean() 
    df['std_rolling'] = df.iloc[:,4].rolling(window=20).mean()
    df = df.dropna()


    # plot rolling avgs or raw mean/std
    col_name = "mean_rolling"  # or mean
    std_name = "std_rolling" # or std

    # y_axis_limits = [[0.0008, 0.003], [0.004, 0.0075], [0, 0.0125]] 
    palette = sns.color_palette("bright", 1)
    for dataset_idx, dataset in enumerate(data.keys()):
        df_ = df[df['dataset'] == dataset]
        ax.set_title(dataset)
        # Plot mean values
        sns.lineplot(data=df_, x="idx", y=col_name, hue="scheduler", ax=ax, linewidth=2.5, palette=palette) # alpha=0.95
        # Manually plot error bounds
        for scheduler_idx, scheduler in enumerate(data[dataset].keys()):
            df__ = df_[df_['scheduler'] == scheduler]
            x = df__['idx'].values
            lower = df__[col_name].values - df__[std_name].values
            upper = df__[col_name].values + df__[std_name].values
            ax.plot(x, lower, color=palette[scheduler_idx], alpha=0.2)
            ax.plot(x, upper, color=palette[scheduler_idx], alpha=0.2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.fill_between(x, lower, upper, alpha=0.1)

        # Set axis limits, render a grid, and label axes
        # ax.set_ylim(y_axis_limits) 
        ax.set_xlim([0, None])

        ax.grid()
        ax.set_xlabel('Minibatch')
        ax.set_ylabel('MSE Loss')

    plt.show()
    exit(0)

if __name__ == "__main__":
    plot_func(config)