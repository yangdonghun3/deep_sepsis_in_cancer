import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ann(nn.Module):
    def __init__(self,in_size, h_size):
        super(ann,self).__init__()
        self.in_size = in_size
        self_h_size = h_size
        self.layer_1 = nn.Linear(in_size, h_size)
        self.out_layer = nn.Linear(h_size, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.batchnorm1 = nn.BatchNorm1d(h_size)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.sigmoid(self.out_layer(x))      
        return x
        
class SkipConnection(nn.Module):
    def __init__(self, in_channels, out_channels, activation, batch_norm, stride):
        super(SkipConnection, self).__init__()

        self.batch_norm = batch_norm
        self.activation = activation
        
        if in_channels != out_channels or stride != 1:
            self.adjust = True
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            if batch_norm:
                self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.adjust = False

    def forward(self, x):
        if self.adjust:
            out = self.conv(x)
            if self.batch_norm:
                out = self.bn(out)
            return self.activation(out)
        else:
            return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, batch_norm, stride=1):
        super(BasicBlock, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if batch_norm:
            self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = SkipConnection(in_channels,out_channels, activation, batch_norm, stride)
        
    def forward(self, x):
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out = self.activation(out)
        out = out + self.shortcut(x)
        return out

class ResNet10(nn.Module):
    def __init__(self, block=BasicBlock, activation=F.relu, batch_norm=True, num_classes=10):
        super(ResNet10, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer = nn.Sequential(
            block(16, 16, activation, batch_norm, stride=1), 
            block(16, 32, activation, batch_norm, stride=2), 
            block(32, 64, activation, batch_norm, stride=2), 
            block(64, 128, activation, batch_norm, stride=2)
            )
        self.linear = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.linear(out)
        out = self.sigmoid(out)
        return out 

class lstm(nn.Module):
    
    def __init__(self,in_size,h_size):
        super(lstm, self).__init__()
        self.in_size=in_size
        self.h_size=h_size
        self.rnn = nn.LSTM(input_size=self.in_size, hidden_size=self.h_size, dropout=0.3)
        #self.linear1 = nn.Linear(in_size, em_size)
        self.linear2 = nn.Linear(h_size,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        #x=self.linear1(x)
        x = x.transpose(0, 1)
        outputs, hidden = self.rnn(x, hidden)        
        outputs = outputs[-1]
        outputs = self.linear2(outputs)
        outputs = self.sigmoid(outputs)

        return outputs