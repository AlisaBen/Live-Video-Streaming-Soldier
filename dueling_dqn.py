from mxnet import nd, init
from mxnet.gluon import nn
# from config import *
import math
import mxnet as mx

class DuelingDQN(nn.Sequential):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        with self.name_scope():
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(channels=32, kernel_size=8, strides=4, activation='relu'),
                nn.Conv2D(channels=64, kernel_size=4, strides=2, activation='relu'),
                nn.Conv2D(channels=64, kernel_size=3, strides=1, activation='relu'),
                nn.Flatten()
            )
            self.fully_connected = nn.Dense(512, activation='relu')
            self.advantage = nn.Dense(8)
            self.value = nn.Dense(1)


    def forward(self, x):
        conv = self.net(x)
        fc = self.fully_connected(conv)
        value = self.value(fc)
        advantage = self.advantage(fc)
        output = value + (advantage - nd.mean(advantage, axis=1, keepdims=True))
        return output


class OriginDQN(nn.Sequential):
    def __init__(self):
        super(OriginDQN, self).__init__()
        with self.name_scope():
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv1D(channels=64,kernel_size=2,strides=1,activation='relu'),
                # nn.Conv2D(channels=64, kernel_size=2, strides=1, activation='relu'),
                # nn.Conv2D(channels=64, kernel_size=4, strides=2, activation='relu'),
                # nn.Conv2D(channels=64, kernel_size=3, strides=1, activation='relu'),
                nn.Flatten()
            )

            self.fully_connected = nn.Dense(512, activation='relu')
            self.value = nn.Dense(8)


    def forward(self, x):
        conv = self.net(x)
        fc = self.fully_connected(conv)
        output = self.value(fc)
        return output
