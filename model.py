1  #!/usr/bin/env python
2  # _*_ coding: utf-8 _*_
3  # @Time : 2020/4/6 11:30 
4  # @Author :Jiwei Tian
5  # @Version：V 0.1
6  # @File : model.py
7  # @desc :
import logging

logging.basicConfig(level=logging.DEBUG, filename='test.log', format="%(levelname)s:%(asctime)s:%(message)s")

import torch.nn.functional as F
import torch.nn as nn



class Model(nn.Module):
    def __init__(self,n_inputs):
        super().__init__()
        self.hidden1 = nn.Linear(n_inputs, 20)
        self.hidden2 = nn.Linear(20, 20)
        self.hidden3 = nn.Linear(20, 40)
        self.hidden4 = nn.Linear(40, 20)
        self.hidden5 = nn.Linear(20, 2)


    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.relu(x)
        x = self.hidden4(x)
        x = F.relu(x)
        x = self.hidden5(x)
        # x = nn.Softmax(x)

        return x



class Model_30bus(nn.Module):
    def __init__(self,n_inputs):
        super().__init__()
        self.hidden1 = nn.Linear(n_inputs, 60)
        self.hidden2 = nn.Linear(60, 60)
        self.hidden3 = nn.Linear(60, 40)
        self.hidden4 = nn.Linear(40, 40)
        self.hidden5 = nn.Linear(40, 20)
        self.hidden6 = nn.Linear(20, 20)
        self.hidden7 = nn.Linear(20, 2)


    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.relu(x)
        x = self.hidden4(x)
        x = F.relu(x)
        x = self.hidden5(x)
        x = F.relu(x)
        x = self.hidden6(x)
        x = F.relu(x)
        x = self.hidden7(x)
        # x = nn.Softmax(x)

        return x

class Model_118bus(nn.Module):
    def __init__(self,n_inputs):
        super().__init__()
        self.hidden1 = nn.Linear(n_inputs, 250)
        self.hidden2 = nn.Linear(250, 120)
        self.hidden3 = nn.Linear(120, 60)
        self.hidden4 = nn.Linear(60, 60)
        self.hidden5 = nn.Linear(60, 40)
        self.hidden6 = nn.Linear(40, 40)
        self.hidden7 = nn.Linear(40, 20)
        self.hidden8 = nn.Linear(20, 20)
        self.hidden9 = nn.Linear(20, 2)


    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.relu(x)
        x = self.hidden4(x)
        x = F.relu(x)
        x = self.hidden5(x)
        x = F.relu(x)
        x = self.hidden6(x)
        x = F.relu(x)
        x = self.hidden7(x)
        x = F.relu(x)
        x = self.hidden8(x)
        x = F.relu(x)
        x = self.hidden9(x)
        # x = nn.Softmax(x)

        return x



class Model2(nn.Module):
    def __init__(self,n_inputs):
        super().__init__()
        self.hidden1 = nn.Linear(n_inputs, 20)
        self.hidden2 = nn.Linear(20, 40)
        self.hidden3 = nn.Linear(40, 20)
        self.hidden4 = nn.Linear(20, 2)


    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.relu(x)
        x = self.hidden4(x)
        # x = nn.Softmax(x)

        return x



class Model2_30bus(nn.Module):
    def __init__(self,n_inputs):
        super().__init__()
        self.hidden1 = nn.Linear(n_inputs, 60)
        self.hidden2 = nn.Linear(60, 40)
        self.hidden3 = nn.Linear(40, 40)
        self.hidden4 = nn.Linear(40, 20)
        self.hidden5 = nn.Linear(20, 20)
        self.hidden6 = nn.Linear(20, 2)


    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.relu(x)
        x = self.hidden4(x)
        x = F.relu(x)
        x = self.hidden5(x)
        x = F.relu(x)
        x = self.hidden6(x)
        # x = nn.Softmax(x)

        return x

class Model2_118bus(nn.Module):
    def __init__(self,n_inputs):
        super().__init__()
        self.hidden1 = nn.Linear(n_inputs, 250)
        self.hidden2 = nn.Linear(250, 120)
        self.hidden3 = nn.Linear(120, 60)
        self.hidden4 = nn.Linear(60, 40)
        self.hidden5 = nn.Linear(40, 40)
        self.hidden6 = nn.Linear(40, 20)
        self.hidden7 = nn.Linear(20, 20)
        self.hidden8 = nn.Linear(20, 2)


    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.relu(x)
        x = self.hidden4(x)
        x = F.relu(x)
        x = self.hidden5(x)
        x = F.relu(x)
        x = self.hidden6(x)
        x = F.relu(x)
        x = self.hidden7(x)
        x = F.relu(x)
        x = self.hidden8(x)
        # x = nn.Softmax(x)

        return x