import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Critic(nn.Module):
    def __init__(self,state_dim,act_dim):
        super(Critic,self).__init__()
        self.fc1=nn.Linear(state_dim,64)
        self.fc2=nn.Linear(64+act_dim,32)
        self.fc3=nn.Linear(32,1)
        
    def forward(self,state,action):
        x=self.fc1(state)
        x=F.relu(x)
        x=self.fc2(th.cat((x,action),1))
        x=F.relu(x)
        x=self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self,state_dim,act_dim):
        super(Actor,self).__init__()
        self.fc1=nn.Linear(state_dim,64)
        self.fc2=nn.Linear(64,32)
        self.fc3=nn.Linear(32,act_dim)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return th.tanh(x)
