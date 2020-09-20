import gym
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

lr=0.001
gamma=0.9
hidden=32

env=gym.make('CartPole-v0')
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


class critic(nn.Module):     #Q net
    def __init__(self):
        super(critic,self).__init__()
        self.fc1=nn.Linear(n_state,hidden)
        self.fc2=nn.Linear(hidden,1)
    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        return x

class AC():
    def __init__(self):
        self.actor=actor()
        self.critic=critic()

        self.Aoptimizer=th.optim.Adam(self.actor.parameters(),lr=lr)
        self.Coptimizer=th.optim.Adam(self.critic.parameters(),lr=lr)

    def choose_action(self,s):
        s=th.FloatTensor(s)
        a_prob=self.actor(s)
        rand=np.random.uniform()
        accumulation=0
        action=0
        for i in range(n_action):
            accumulation+=a_prob[i]
            if accumulation>=rand:
                action=i
                break
        return action

    def actor_learn(self,s,a,td_error):
        s=th.FloatTensor(s)
        a_prob=self.actor(s)[a]
        loss=-(th.log(a_prob)*td_error.detach())

        self.Aoptimizer.zero_grad()
        loss.backward()
        self.Aoptimizer.step()

    def critic_learn(self,transition):    #transition=[s,[r],[a],s_,[done]]
        s=th.FloatTensor(transition[0])
        r=transition[1][0]
        s_=th.FloatTensor(transition[3])
        done=transition[4][0]

        v_eval=self.critic(s)
        v_target=self.critic(s_)*gamma+r

        td_error=v_eval-v_target.detach()
        loss=td_error**2

        self.Coptimizer.zero_grad()
        loss.backward()
        self.Coptimizer.step()
        return td_error
    

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

        td_error=ac.critic_learn(transition)
        ac.actor_learn(s,a,td_error)
        if done:
            break
        s=s_
    if(episode%10==0):
        print("Episode:"+format(episode)+",score:"+format(total_reward))


        


