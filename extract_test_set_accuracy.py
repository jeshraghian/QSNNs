import numpy as np
import pandas as pd
import os


data = {
    4: {
        "MNIST": {
            "cosine": [
                "mnist/acc_MNIST_t0.csv",
                "mnist/acc_MNIST_t1.csv",
                "mnist/acc_MNIST_t2.csv",
            ],
        },
        "FashionMNIST": {
            "cosine": [
                "fmnist/acc_FMNIST_t0.csv",
                "fmnist/acc_FMNIST_t1.csv",
                "fmnist/acc_FMNIST_t2.csv",
            ],
        },
        "DVS128 Gesture": {
            "cosine": [
                "DVS/acc_DVS_t0.csv",
                "DVS/acc_DVS_t1.csv",
                "DVS/acc_DVS_t2.csv",
            ],
        },
    },
}


df = pd.DataFrame(
    columns=[
        "dataset",
        "network_precision",
        "scheduler",
        "test_set_accuracy_best",
        "test_set_accuracy_mean",
        "test_set_accuracy_std",
    ]
)
for precision in data.keys():
    for dataset_idx, dataset in enumerate(data[precision].keys()):
        for scheduler in data[precision][dataset].keys():
            test_set_accuracy_values = []
            for idx, trial in enumerate(data[precision][dataset][scheduler]):
                trial_df = pd.read_csv(trial)
                test_set_accuracy = trial_df["test_acc"].max().item()
                test_set_accuracy_values.append(test_set_accuracy)

            df = df.append(
                {
                    "dataset": dataset,
                    "network_precision": precision,
                    "scheduler": scheduler,
                    "test_set_accuracy_best": max(test_set_accuracy_values),
                    "test_set_accuracy_mean": np.mean(test_set_accuracy_values),
                    "test_set_accuracy_std": np.std(test_set_accuracy_values),
                },
                ignore_index=True,
            )

df.to_csv("test_set_accuracy.csv", index=False)
