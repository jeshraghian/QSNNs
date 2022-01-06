# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# misc
import pandas as pd
import time

# local imports
from dataloader import *
from Net import *
from test_acc import *
from train import *
from earlystopping import *
from conf import *
from plot_loss import *

####################################################
##     Modify config in conf to reparameterize    ##
####################################################

file_name = config["exp_name"]

for trial in range(config["num_trials"]):

    # file names
    SAVE_CSV = config["save_csv"]
    SAVE_MODEL = config["save_model"]
    csv_name = file_name + "_t" + str(trial) + ".csv"
    log_name = file_name + "_t" + str(trial) + ".log"
    model_name = file_name + "_t" + str(trial) + ".pt"
    num_epochs = config["num_epochs"]

    # dataframes
    df_train_loss = pd.DataFrame()
    df_test_acc = pd.DataFrame(columns=["epoch", "test_acc", "train_time"])
    df_lr = pd.DataFrame()

    # initialize network
    net = Net(config)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    # net params
    criterion = SF.mse_count_loss(
        correct_rate=config["correct_rate"], incorrect_rate=config["incorrect_rate"]
    )
    optimizer, scheduler = optim_func(net, config)

    # early stopping condition
    if config["early_stopping"]:
        early_stopping = EarlyStopping_acc(
            patience=config["patience"], verbose=True, path=model_name
        )
        early_stopping.early_stop = False
        early_stopping.best_score = None

    # load data
    trainset, testset = load_data(config)
    config["dataset_length"] = len(trainset)
    trainloader = DataLoader(
        trainset, batch_size=int(config["batch_size"]), shuffle=True
    )
    testloader = DataLoader(
        testset, batch_size=int(config["batch_size"]), shuffle=False
    )

    print(
        f"=======Trial: {trial}, Quantization: {config['num_bits']}, beta: {config['beta']:.3f}, threshold: {config['threshold']:.2f}, slope: {config['slope']}, lr: {config['lr']:.3e}======"
    )

    # training loop
    for epoch in range(num_epochs):

        # train
        start_time = time.time()
        loss_list, lr_list = train(
            config, net, trainloader, criterion, optimizer, device, scheduler
        )
        epoch_time = time.time() - start_time

        # test
        test_acc = test_accuracy(config, net, testloader, device)
        print(f"Epoch: {epoch} \tTest Accuracy: {test_acc}")

        # record data / save as csv
        df_lr = df_lr.append(lr_list, ignore_index=True)
        df_train_loss = df_train_loss.append(loss_list, ignore_index=True)
        df_test_acc = df_test_acc.append(
            {"epoch": epoch, "test_acc": test_acc, "train_time": epoch_time},
            ignore_index=True,
        )

        if SAVE_CSV:
            df_train_loss.to_csv("loss_" + csv_name, index=False)
            df_test_acc.to_csv("acc_" + csv_name, index=False)
            df_lr.to_csv("lr_" + csv_name, index=False)

        if config["early_stopping"]:
            early_stopping(test_acc, net)

            if early_stopping.early_stop:
                print("Early stopping")
                early_stopping.early_stop = False
                early_stopping.best_score = None
                break

        if SAVE_MODEL and not config["early_stopping"]:
            torch.save(net.state_dict(), model_name)

if config["plot_loss"]:
    plot_func(config)
