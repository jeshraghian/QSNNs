import torch
import torch.nn as nn


def train(config, net, trainloader, criterion, optimizer, device="cpu", scheduler=None):
    net.train()
    loss_accum = []
    lr_accum = []
    i = 0
    for data, labels in trainloader:
        data, labels = data.to(device), labels.to(device)
        spk_rec, _ = net(data)
        loss = criterion(spk_rec, labels)
        optimizer.zero_grad()
        loss.backward()
        if config["grad_clip"]:
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)

        if config["weight_clip"]:
            with torch.no_grad():
                for param in net.parameters():
                    param.clamp_(-1, 1)

        optimizer.step()
        scheduler.step()
        loss_accum.append(loss.item() / config["num_steps"])
        lr_accum.append(optimizer.param_groups[0]["lr"])

    return loss_accum, lr_accum
