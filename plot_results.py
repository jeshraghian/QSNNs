import numpy as np
from numpy.lib.twodim_base import tri
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["figure.figsize"] = (32.5, 10)
plt.rcParams.update({"font.size": 18})
plt.rcParams["axes.linewidth"] = 2
plt.rcParams["axes.formatter.limits"] = [-5, 4]

fig, ax = plt.subplots(1, 3)

data = {
    4: {
        "MNIST": {
            "cosine": [
                "mnist/loss_MNIST_t0.csv",
                "mnist/loss_MNIST_t1.csv",
                "mnist/loss_MNIST_t2.csv",
            ],
        },
        "FashionMNIST": {
            "cosine": [
                "fmnist/loss_FMNIST_t0.csv",
                "fmnist/loss_FMNIST_t1.csv",
                "fmnist/loss_FMNIST_t2.csv",
            ],
        },
        "DVS128 Gesture": {
            "cosine": [
                "DVS/loss_DVS_t0.csv",
                "DVS/loss_DVS_t1.csv",
                "DVS/loss_DVS_t2.csv",
            ],
        },
    },
}

df = pd.DataFrame(
    columns=["dataset", "network_precision", "scheduler", "idx", "mean", "std"]
)
for precision in data.keys():
    for dataset_idx, dataset in enumerate(data[precision].keys()):
        for scheduler in data[precision][dataset].keys():
            grouped_trial_df = pd.DataFrame(columns=["idx", "loss"])
            for idx, trial in enumerate(data[precision][dataset][scheduler]):
                if trial is not None:
                    trial_df = pd.read_csv(trial)
                    trial_data = np.vstack(
                        (trial_df.index, trial_df.values.flatten())
                    ).T
                    trial_df = pd.DataFrame(trial_data, columns=["idx", "loss"])
                    grouped_trial_df = grouped_trial_df.append(trial_df)
                else:
                    grouped_trial_df = grouped_trial_df.append(
                        {"idx": 0, "loss": 1}, ignore_index=True
                    )

            grouped_trial_df["loss"] = pd.to_numeric(grouped_trial_df["loss"])
            grouped_trial_df_ = grouped_trial_df.groupby("idx")
            grouped_trial_data = np.hstack(
                (
                    np.expand_dims(grouped_trial_df["idx"].unique(), 1),
                    grouped_trial_df_.mean(),
                    grouped_trial_df_.std(),
                )
            )
            grouped_trial_data = np.nan_to_num(grouped_trial_data, nan=0)
            trial_df = pd.DataFrame(grouped_trial_data, columns=["idx", "mean", "std"])
            trial_df["network_precision"] = precision
            trial_df["scheduler"] = scheduler
            trial_df["dataset"] = dataset
            df = df.append(trial_df, ignore_index=True)

df = df[df["idx"] % 250 == 0]
df.to_csv("loss_data.csv")
df_quant = df[df["network_precision"] == 4]
del df
# Separate out dataframes to independently take moving avgs
df_mnist_quant = df_quant[df_quant["dataset"] == "MNIST"]
df_fmnist_quant = df_quant[df_quant["dataset"] == "FashionMNIST"]
df_dvs_quant = df_quant[df_quant["dataset"] == "DVS128 Gesture"]
del df_quant
df_mnist_quant["mean_rolling"] = (
    df_mnist_quant.iloc[:, 4].rolling(window=20, min_periods=1).mean()
)
df_mnist_quant["std_rolling"] = (
    df_mnist_quant.iloc[:, 5].rolling(window=20, min_periods=1).mean()
)
df_mnist_quant = df_mnist_quant.dropna()
df_fmnist_quant["mean_rolling"] = (
    df_fmnist_quant.iloc[:, 4].rolling(window=20, min_periods=1).mean()
)
df_fmnist_quant["std_rolling"] = (
    df_fmnist_quant.iloc[:, 5].rolling(window=20, min_periods=1).mean()
)
df_fmnist_quant = df_fmnist_quant.dropna()
df_dvs_quant["mean_rolling"] = (
    df_dvs_quant.iloc[:, 4].rolling(window=5, min_periods=1).mean()
)
df_dvs_quant["std_rolling"] = (
    df_dvs_quant.iloc[:, 5].rolling(window=5, min_periods=1).mean()
)
df_dvs_quant = df_dvs_quant.dropna()
# Combine them
frames = [df_mnist_quant, df_fmnist_quant, df_dvs_quant]
df = pd.concat(frames, ignore_index=True)
# Plot rolling avgs or raw mean/std
col_name = "mean_rolling"  # or mean
std_name = "std_rolling"  # or std
y_axis_limits = [[0.0008, 0.003], [0.004, 0.0075], [0.001, 0.0125]]
palette = sns.color_palette("bright", 4)
for precision_idx, precision in enumerate(data.keys()):
    for dataset_idx, dataset in enumerate(data[precision].keys()):
        df_tmp = df[df["network_precision"] == precision]
        df_ = df_tmp[df_tmp["dataset"] == dataset]
        # Plot mean values
        sns.lineplot(
            data=df_,
            x="idx",
            y=col_name,
            hue="scheduler",
            ax=ax[dataset_idx],
            linewidth=2.5,
            palette=palette,
        )  # alpha=0.95
        # Manually plot error bounds
        for scheduler_idx, scheduler in enumerate(data[precision][dataset].keys()):
            df__ = df_[df_["scheduler"] == scheduler]
            x = df__["idx"].values
            try:
                lower = df__[col_name].values - df__[std_name].values
                upper = df__[col_name].values + df__[std_name].values
                ax[dataset_idx].plot(x, lower, color=palette[scheduler_idx], alpha=0.2)
                ax[dataset_idx].plot(x, upper, color=palette[scheduler_idx], alpha=0.2)
                ax[dataset_idx].spines["top"].set_visible(False)
                ax[dataset_idx].spines["right"].set_visible(False)
                ax[dataset_idx].fill_between(x, lower, upper, alpha=0.1)
            except:
                pass

        ax[dataset_idx].set_title(dataset)
        ax[dataset_idx].set_xlim([0, None])
        ax[dataset_idx].set_ylim(y_axis_limits[dataset_idx])
        ax[dataset_idx].yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
        ax[dataset_idx].grid()
        ax[dataset_idx].set_xlabel("Minibatch")
        ax[dataset_idx].set_ylabel("MSE Loss")
        print(precision, dataset)

plt.show()
