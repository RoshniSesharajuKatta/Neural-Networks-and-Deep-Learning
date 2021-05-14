# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # INSERT CODE HERE
        self.hidden_layer = nn.Linear(2,num_hid, bias = True)
        self.output_layer = nn.Linear(num_hid,1, bias = True)

    def forward(self, input):
        x = input[:, 0]
        y = input[:, 1]
        r = torch.sqrt((x*x + y*y))
        r = r.reshape(-1,1)
        a = torch.atan2(y, x)
        a = a.reshape(-1,1)
        #concatenating the two tensors
        concatenated_tensor = torch.cat(tensors = (r, a), dim = 1) 
        self.hidden_activation1 = torch.tanh(self.hidden_layer(concatenated_tensor))
        output = torch.sigmoid(self.output_layer(self.hidden_activation1))
        #output = 0*input[:,0] # CHANGE CODE HERE
        return output

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # INSERT CODE HERE
        self.hidden1 = nn.Linear(2, num_hid, bias = True)
        self.hidden2 = nn.Linear(num_hid, num_hid, bias = True)
        self.output_layer3 = nn.Linear(num_hid, 1, bias = True)

    def forward(self, input):
        self.hidden_activation1 = torch.tanh(self.hidden1(input))
        self.hidden_activation2 = torch.tanh(self.hidden2(self.hidden_activation1))
        output = torch.sigmoid(self.output_layer3(self.hidden_activation2))
        #output = 0*input[:,0] # CHANGE CODE HERE
        return output

def graph_hidden(net, layer, node):
    plt.clf()
    # INSERT CODE HERE
    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): 
        net.eval()        
        output = net(grid)
        
        #For the polar net, there is only one hidden layer. Raw net has 2 hidden layers. 
        #When there is only 1 hidden layer, this is run otherwise the eslse part of the code runs.
        #For PolarNet and RawNet  
        if layer == 1:
          pred = (net.hidden_activation1[:, node]>=0).float()
        #For RawNet
        else:
          pred = (net.hidden_activation2[:, node]>=0).float()
        
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')
