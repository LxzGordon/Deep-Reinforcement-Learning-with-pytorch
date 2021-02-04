import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Q(nn.Module):
    def __init__(self,state_dim,act_dim):
        super(Q,self).__init__()
        self.fc1=nn.Linear(state_dim,64)
        self.fc2=nn.Linear(64+act_dim,256)
        self.fc3=nn.Linear(256,1)
        
    def forward(self,state,action):
        x=self.fc1(state)
        x=F.relu(x)
        x=self.fc2(th.cat((x,action),1))
        x=F.relu(x)
        x=self.fc3(x)
        return x

class V(nn.Module):
    def __init__(self,state_dim):
        super(V,self).__init__()
        self.fc1=nn.Linear(state_dim,256)
        self.fc2=nn.Linear(256,1)
        
    def forward(self,state):
        x=self.fc1(state)
        x=F.relu(x)
        x=self.fc2(x)
        return x

class Actor(nn.Module):
    def __init__(self,state_dim,max_a):
        super(Actor,self).__init__()
        self.fc1=nn.Linear(state_dim,256)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2=nn.Linear(256,64)
        self.fc2.weight.data.normal_(0, 0.1)
        self.std_head=nn.Linear(64,1)
        self.std_head.weight.data.normal_(0, 0.1)
        self.mu_head=nn.Linear(64,1)
        self.mu_head.weight.data.normal_(0, 0.1)
        self.max_a=max_a

        self.max_std=2
        self.min_std=-2

    def forward(self,x):
        x=self.fc2(F.relu(self.fc1(x)))
        std=self.std_head(x)
        mu=self.mu_head(x)
        std= th.clamp(std, self.min_std, self.max_std)
        return mu,std