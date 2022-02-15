import snntorch as snn
from snntorch import surrogate
import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn


class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_bits = config["num_bits"]
        self.thr = config["threshold"]
        self.slope = config["slope"]
        self.beta = config["beta"]
        self.num_steps = config["num_steps"]
        self.batch_norm = config["batch_norm"]
        self.p1 = config["dropout"]
        self.spike_grad = surrogate.fast_sigmoid(self.slope)
        if self.num_bits is None:
            self.init_net()
        else:
            self.init_quantized_net()

    def init_net(self):
        self.conv1 = nn.Conv2d(1, 16, 5, bias=False)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.lif1 = snn.Leaky(self.beta, threshold=self.thr, spike_grad=self.spike_grad)
        self.conv2 = nn.Conv2d(16, 64, 5, bias=False)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.lif2 = snn.Leaky(self.beta, threshold=self.thr, spike_grad=self.spike_grad)
        self.fc1 = nn.Linear(64 * 4 * 4, 10)
        self.lif3 = snn.Leaky(self.beta, threshold=self.thr, spike_grad=self.spike_grad)
        self.dropout = nn.Dropout(self.p1)

    def init_quantized_net(self):
        self.conv1 = qnn.QuantConv2d(
            1, 16, 5, bias=False, weight_bit_width=self.num_bits
        )
        self.conv1_bn = nn.BatchNorm2d(16)
        self.lif1 = snn.Leaky(self.beta, threshold=self.thr, spike_grad=self.spike_grad)
        self.conv2 = qnn.QuantConv2d(
            16, 64, 5, bias=False, weight_bit_width=self.num_bits
        )
        self.conv2_bn = nn.BatchNorm2d(64)
        self.lif2 = snn.Leaky(self.beta, threshold=self.thr, spike_grad=self.spike_grad)
        self.fc1 = qnn.QuantLinear(
            64 * 4 * 4, 10, bias=False, weight_bit_width=self.num_bits
        )
        self.lif3 = snn.Leaky(self.beta, threshold=self.thr, spike_grad=self.spike_grad)
        self.dropout = nn.Dropout(self.p1)

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        # Record the final layer
        spk3_rec = []
        mem3_rec = []
        for step in range(self.num_steps):
            cur1 = F.avg_pool2d(self.conv1(x), 2)
            if self.batch_norm:
                cur1 = self.conv1_bn(cur1)

            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = F.avg_pool2d(self.conv2(spk1), 2)
            if self.batch_norm:
                cur2 = self.conv2_bn(cur2)

            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.dropout(self.fc1(spk2.flatten(1)))
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)
