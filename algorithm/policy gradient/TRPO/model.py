import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class net(nn.Module):
    def __init__(self,n_state,n_action):
        super(net,self).__init__()
        self.fc1=nn.Linear(n_state,256)
        self.fc2=nn.Linear(256,n_action)
        self.softmax=nn.Softmax()
    def forward(self,s):
        s=th.relu(self.fc1(s))
        return self.softmax(self.fc2(s))


class v(nn.Module):
    def __init__(self,n_state):
        super(v,self).__init__()
        self.fc1=nn.Linear(n_state,256)
        self.fc2=nn.Linear(256,1)
    def forward(self, s):
        return self.fc2(F.relu(self.fc1(s)))
