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

class REINFORCE():
    def __init__(self):
        self.policy_net=policy()
        self.g=0
        self.optimizer=th.optim.Adam(self.policy_net.parameters(),lr=alpha)
    def choose_action(self,s):
        s=th.FloatTensor(s)
        a_prob=self.policy_net(s)
        entropy=-(a_prob*a_prob.log()).sum()
        rand=np.random.uniform()
        accumulation=0
        action=0
        for i in range(n_action):
            accumulation+=a_prob[i]
            if accumulation>=rand:
                action=i
                break
        return action,entropy,rand

    def learn(self,transition,entropy):
        timestep=len(transition)
        loss=0
        returns=th.zeros(timestep,1)
        log_prob=th.zeros(timestep,1)
        self.g=0
        for i in reversed(range(timestep)):
            s=th.FloatTensor(transition[i,0])
            a=transition[i,1][0]
            r=transition[i,2][0]
            log_prob[i]=th.log(self.policy_net(s))[a].unsqueeze(0)
            self.g=gamma*self.g+r
            returns[i]=self.g


        loss=-(log_prob*returns.detach()).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

reinforce=REINFORCE()

for episode in range(10000):
    t=0
    s=env.reset()
    transition=np.array([])
    total_reward=0
    entropy=0
    while(t<300):
        a,entro,rand=reinforce.choose_action(s)
        s_,r,done,_=env.step(a)
        total_reward+=r
        trans=[s,[a],[r]]
        if t==0:
            transition=trans
        else:
            transition=np.vstack((transition,trans))
        if done:
            reinforce.learn(transition,entropy)
            break
        s=s_
        t+=1
    if episode%100==0:
        print("Episode:"+format(episode)+",total score:"+format(total_reward))


    
