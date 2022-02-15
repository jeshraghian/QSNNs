import snntorch as snn
from snntorch import functional as SF
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import time
from earlystopping import *
from set_all_seeds import set_all_seeds


def evaluate(Net, config, load_data, train, test, optim_func):
    file_name = config["exp_name"]
    for trial in range(config["num_trials_eval"]):
        csv_name = file_name + "_t" + str(trial) + ".csv"
        model_name = file_name + "_t" + str(trial) + ".pt"
        num_epochs = config["num_epochs_eval"]
        set_all_seeds(config["seed"] + trial)
        df_train_loss = pd.DataFrame()
        df_test_acc = pd.DataFrame(columns=["epoch", "test_acc", "train_time"])
        df_lr = pd.DataFrame()
        # Initialize the network
        net = Net(config)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        net.to(device)
        # Initialize the optimizer and scheduler
        criterion = SF.mse_count_loss(
            correct_rate=config["correct_rate"], incorrect_rate=config["incorrect_rate"]
        )
        optimizer, scheduler, loss_dependent = optim_func(net, config)
        # Early stopping condition
        if config["early_stopping"]:
            early_stopping = EarlyStopping_acc(
                patience=config["patience"], verbose=True, path=model_name
            )
            early_stopping.early_stop = False
            early_stopping.best_score = None

        # Load data
        trainset, testset = load_data(config)
        config["dataset_length"] = len(trainset)
        trainloader = DataLoader(
            trainset, batch_size=int(config["batch_size"]), shuffle=True
        )
        testloader = DataLoader(
            testset, batch_size=int(config["batch_size"]), shuffle=False
        )
        if loss_dependent:
            old_loss_hist = float("inf")

        print(
            f"=======Trial: {trial}, Batch: {config['batch_size']}, beta: {config['beta']:.3f}, threshold: {config['threshold']:.2f}, slope: {config['slope']}, lr: {config['lr']:.3e}======"
        )
        # Train
        for epoch in range(num_epochs):
            start_time = time.time()
            loss_list, lr_list = train(
                config, net, trainloader, criterion, optimizer, device, scheduler
            )
            epoch_time = time.time() - start_time
            if loss_dependent:
                avg_loss_hist = sum(loss_list) / len(loss_list)
                if avg_loss_hist > old_loss_hist:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = param_group["lr"] * 0.5
                else:
                    old_loss_hist = avg_loss_hist

            # Test
            test_accuracy = test(config, net, testloader, device)
            print(f"Epoch: {epoch} \tTest Accuracy: {test_accuracy}")
            df_lr = df_lr.append(lr_list, ignore_index=True)

            df_train_loss = df_train_loss.append(loss_list, ignore_index=True)
            df_test_acc = df_test_acc.append(
                {"epoch": epoch, "test_acc": test_accuracy, "train_time": epoch_time},
                ignore_index=True,
            )
            if config["save_csv"]:
                df_train_loss.to_csv("loss_" + csv_name, index=False)
                df_test_acc.to_csv("acc_" + csv_name, index=False)
                df_lr.to_csv("lr_" + csv_name, index=False)

            if config["early_stopping"]:
                early_stopping(test_accuracy, net)
                if early_stopping.early_stop:
                    print("Early stopping")
                    early_stopping.early_stop = False
                    early_stopping.best_score = None
                    break
