import gym
import torch as th
import numpy as np
from model import Q,V,Actor
from experience_replay import replay_memory
from torch.distributions.normal import Normal

lr=0.001
tau=0.005
max_t=200
gamma=0.9
memory_size=2000
warmup=500
batchsize=32
env=gym.make('Pendulum-v0')
device="cuda"
alpha=0.05
n_action=1
n_state=env.observation_space.shape[0]
max_action = float(env.action_space.high[0])

class SAC():
    def __init__(self):
        self.V=V(n_state).to(device)
        self.target_V=V(n_state).to(device)
        self.policy=Actor(n_state,max_action).to(device)
        self.Q=Q(n_state,n_action).to(device)
        
        self.optimV=th.optim.Adam(self.V.parameters(),lr=lr)
        self.optimQ=th.optim.Adam(self.Q.parameters(),lr=lr)
        self.optimP=th.optim.Adam(self.policy.parameters(),lr=lr)

        self.memory=replay_memory(memory_size)

    def choose_action(self,s):
        mu,log_std=self.policy(s)

        dist=Normal(mu,th.exp(log_std))
        action=dist.sample()
        action = th.tanh(action)
        
        return action


    def V_learn(self,batch):
        b_s=th.FloatTensor(batch[:,0].tolist()).to(device)
        b_a=th.FloatTensor(batch[:,2].tolist()).to(device)

        mu,log_std=self.policy(b_s)
        dist=Normal(mu,th.exp(log_std))

        z=dist.sample()
        b_a=th.tanh(z)
        prob=dist.log_prob(z)
        qs=self.Q(b_s,b_a)

        v=self.V(b_s)
        target_v=qs-prob

        loss=(v-target_v.detach())**2
        loss=loss.mean()

        self.optimV.zero_grad()
        loss.backward()
        self.optimV.step()



    def Q_learn(self,batch):
        b_s=th.FloatTensor(batch[:,0].tolist()).to(device)
        b_r=th.FloatTensor(batch[:,1].tolist()).to(device)
        b_a=th.FloatTensor(batch[:,2].tolist()).to(device)
        b_s_=th.FloatTensor(batch[:,3].tolist()).to(device)
        b_d=th.FloatTensor(batch[:,4].tolist()).to(device)

        target_q=b_r+(1-b_d)*gamma*self.target_V(b_s_)

        eval_q=self.Q(b_s,b_a)
        loss=(eval_q-target_q.detach())**2
        loss=loss.mean()
        self.optimQ.zero_grad()
        loss.backward()
        self.optimQ.step()
        

    def P_learn(self,batch):
        b_s=th.FloatTensor(batch[:,0].tolist()).to(device)

        norm=Normal(th.zeros((batchsize,1)),th.ones((batchsize,1)))
        #norm=Normal(0,1)
        mu,log_std=self.policy(b_s)

        z=norm.sample()
        b_a=th.tanh(mu+th.exp(log_std)*z.to(device))

        dist=Normal(mu,th.exp(log_std))
        log_prob=dist.log_prob(mu+th.exp(log_std)*z.to(device))- th.log(1 - b_a.pow(2) + 1e-7)
        qs=self.Q(b_s,b_a)

        loss=alpha*log_prob-qs
        loss=loss.mean()

        self.optimP.zero_grad()
        loss.backward()
        self.optimP.step()


    def soft_update(self):
        for param,target_param in zip(self.V.parameters(),self.target_V.parameters()):
            target_param.data.copy_(tau*param.data+(1-tau)*target_param.data)


sac=SAC()
def main():
    for episode in range(2000):
        s=env.reset()
        total_reward=0
        t=0
        while t<max_t:
            a=sac.choose_action(th.FloatTensor(s).to(device))
            s_,r,done,_=env.step([a.tolist()[0]*max_action])
            total_reward+=r
            transition=[s,[r],[a],s_,[done]]
            sac.memory.store(transition)
            if done:
                break
            s=s_
            if(sac.memory.size()<warmup):
                continue
            batch=sac.memory.sample(batchsize)
            sac.V_learn(batch)
            sac.Q_learn(batch)
            sac.P_learn(batch)
            sac.soft_update()
            t+=1
        print("episode:"+format(episode)+",test score:"+format(total_reward))

if __name__=='__main__':
    main()
