# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        #linear layer
        self.lin1 = nn.Linear(28*28, 10)
        

    def forward(self, x):
        x = x.view(x.shape[0],-1)
        output = F.log_softmax(self.lin1(x), dim=1)
        return output # CHANGE CODE HERE

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        #two layers
        self.input_to_hidden_layer = nn.Linear(28*28, 300)
        self.hidden_to_output_layer = nn.Linear(300, 10)
        # INSERT CODE HERE

    def forward(self, x):
        x = x.view(x.shape[0],-1)
        #lin_sum_at_hidden_layer = self.input_to_hidden_layer(self.input_to_hidden_layer(x))
        hidden_layer_tanh = torch.tanh(self.input_to_hidden_layer(x))
        #lin_sum_at_output_layer = self.hidden_to_output_layer(self.hidden_to_output_layer(x))
        output = F.log_softmax(self.hidden_to_output_layer(hidden_layer_tanh), dim=1)
        return output # CHANGE CODE HERE

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.layer_1 = nn.Sequential(
          nn.Conv2d(1, 72, kernel_size = 5),
          nn.MaxPool2d(kernel_size = 2, stride = 2),
          #nn.ReLu(),
        )
        self.layer_2 = nn.Sequential(
          nn.Conv2d(72, 144, kernel_size = 5),
          nn.MaxPool2d(kernel_size = 2, stride = 2),
          #nn.ReLu()
        )
        #self.conv1 = nn.Conv2d(3, 18, kernel_size = 5)
        #self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        #self.conv2 = nn.Conv2d(18, 36, kernel_size = 5)
        #self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.full_connected_layer = nn.Linear(144*4*4, 300)
        self.lin_layer = nn.Linear(300, 10)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        #x = self.max_pool1(x)
        x = F.relu(self.layer_2(x))
        #x = self.max_pool2(x)
        #x = self.layer_1(x)
        x = x.view(x.shape[0],-1)
        x = F.relu(self.full_connected_layer(x))
        output = F.log_softmax(self.lin_layer(x), dim = 1)
        return output # CHANGE CODE HERE
