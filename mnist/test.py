import torch
from snntorch import functional as SF


def test(config, net, testloader, device="cpu"):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, _ = net(images)
            accuracy = SF.accuracy_rate(outputs, labels)
            total += labels.size(0)
            correct += accuracy * labels.size(0)

    return 100 * correct / total
