import gym
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

alpha=0.001
max_t=200
gamma=0.9
hidden=32
env=gym.make('CartPole-v0')
env=env.unwrapped
n_action=env.action_space.n
n_state=env.observation_space.shape[0]

class policy(nn.Module):
    def __init__(self):
        super(policy,self).__init__()
        self.fc1=nn.Linear(n_state,hidden)
        self.fc2=nn.Linear(hidden,n_action)
        self.softmax=nn.Softmax()
    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        prob=self.softmax(x)
        return prob

class Vnet(nn.Module):
    def __init__(self):
        super(Vnet,self).__init__()
        self.fc1=nn.Linear(n_state,hidden)
        self.fc2=nn.Linear(hidden,1)
    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        out=self.fc2(x)
        return out


class baseline_REINFORCE():
    def __init__(self):
        self.policy_net=policy()
        self.V=Vnet()               #baseline
        self.g=0
        self.optimizerP=th.optim.Adam(self.policy_net.parameters(),lr=alpha)
        self.optimizerV=th.optim.Adam(self.V.parameters(),lr=alpha)
    def choose_action(self,s):
        s=th.FloatTensor(s)
        a_prob=self.policy_net(s)
        rand=np.random.uniform()
        accumulation=0
        action=0
        for i in range(n_action):
            accumulation+=a_prob[i]
            if accumulation>=rand:
                action=i
                break
        return action

    def learn(self,transition):
        timestep=len(transition)
        loss=0
        loss_v=0
        returns=th.zeros(timestep,1)
        log_prob=th.zeros(timestep,1)
        v=th.zeros(timestep,1)
        self.g=0
        for i in reversed(range(timestep)):
            s=th.FloatTensor(transition[i,0])
            a=transition[i,1][0]
            r=transition[i,2][0]
            v[i]=self.V(s)            #calculate b(s)
            log_prob[i]=th.log(self.policy_net(s))[a].unsqueeze(0)
            self.g=gamma*self.g+r
            returns[i]=self.g

        loss=-(log_prob*(returns-v).detach()).sum()
        loss_v=(0.5*(returns-v)**2).sum()

        #update separately
        self.optimizerP.zero_grad()
        loss.backward()
        self.optimizerP.step()

        self.optimizerV.zero_grad()
        loss_v.backward()
        self.optimizerV.step()

reinforce=baseline_REINFORCE()

for episode in range(10000):
    t=0
    s=env.reset()
    transition=np.array([])
    total_reward=0
    while(t<300):
        a=reinforce.choose_action(s)
        s_,r,done,_=env.step(a)
        total_reward+=r
        trans=[s,[a],[r]]
        if t==0:
            transition=trans
        else:
            transition=np.vstack((transition,trans))
        if done:
            reinforce.learn(transition)
            break
        s=s_
        t+=1
    if episode%100==0:
        print("Episode:"+format(episode)+",total score:"+format(total_reward))


    
