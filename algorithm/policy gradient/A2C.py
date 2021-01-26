import gym
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
lr=0.001
gamma=0.9
hidden=32
env=gym.make('CartPole-v0')
device="cuda"
env=env.unwrapped
n_action=env.action_space.n
n_state=env.observation_space.shape[0]

class actor(nn.Module):      #policy net
    def __init__(self):
        super(actor,self).__init__()
        self.fc1=nn.Linear(n_state,hidden)
        self.fc2=nn.Linear(hidden,n_action)
        self.softmax=nn.Softmax()
    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        prob=self.softmax(x)
        return prob


class Q(nn.Module):     
    def __init__(self):
        super(Q,self).__init__()
        self.q=nn.Sequential(nn.Linear(n_state,hidden),
                                       nn.ReLU(),
                             nn.Linear(hidden,n_action))
    def forward(self,x):
        q=self.q(x)
        return q

class V(nn.Module):     
    def __init__(self):
        super(V,self).__init__()
        self.v=nn.Sequential(nn.Linear(n_state,hidden),
                             nn.ReLU(),
                             nn.Linear(hidden,1))
    def forward(self,x):
        v=self.v(x)
        return v

class critic(nn.Module):     
    def __init__(self):
        super(critic,self).__init__()
        self.v=V()
        self.q=Q()

    def forward(self,x):
        v=self.v(x)
        q=self.q(x)
        advantage=q-v.repeat(2)
        return advantage


class AC():
    def __init__(self):
        self.actor=actor().to(device)
        self.critic=critic().to(device)

        self.Aoptimizer=th.optim.Adam(self.actor.parameters(),lr=lr)
        self.Qoptimizer=th.optim.Adam(self.critic.q.parameters(),lr=lr)
        self.Voptimizer=th.optim.Adam(self.critic.v.parameters(),lr=lr)

    def choose_action(self,s):
        s=th.FloatTensor(s).to(device)
        a_prob=self.actor(s)
        dist=Categorical(a_prob)
        action=dist.sample().tolist()
        return action

    def actor_learn(self,s,a,A):
        s=th.FloatTensor(s).to(device)
        a_prob=self.actor(s)[a]
        loss=-(th.log(a_prob)*A.detach())

        self.Aoptimizer.zero_grad()
        loss.backward()
        self.Aoptimizer.step()

    def critic_learn(self,transition):    #transition=[s,[r],[a],s_,[done]]
        s=th.FloatTensor(transition[0]).to(device)
        r=transition[1][0]
        s_=th.FloatTensor(transition[3]).to(device)
        done=transition[4][0]

        a=transition[2][0]
        q=self.critic.q(s)[a]
        v=self.critic.v(s)
        A=q-v
        v_=self.critic.v(s_)*gamma+r
        if not done:
            q_target=th.max(self.critic.q(s_))*gamma+r   
            loss_q=(q-q_target.detach())**2
        else:
            q_target=r
            loss_q=(q-q_target)**2
        loss_v=(v-v_.detach())**2
        #print(loss)
        self.Qoptimizer.zero_grad()
        loss_q.backward()
        self.Qoptimizer.step()
        self.Voptimizer.zero_grad()
        loss_v.backward()
        self.Voptimizer.step()
        return A
    

ac=AC()
    
for episode in range(10000):
    t=0
    s=env.reset()
    total_reward=0
    while(t<300):
        a=ac.choose_action(s)
        s_,r,done,_=env.step(a)
        total_reward+=r
        transition=[s,[r],[a],s_,[done]]

        A=ac.critic_learn(transition)
        ac.actor_learn(s,a,A)
        if done:
            break
        s=s_
    if(episode%10==0):
        print("Episode:"+format(episode)+",score:"+format(total_reward))


        


