import gym
import torch as th
import numpy as np
from model import Critic,Actor
from experience_reply import replay_memory


lr=0.001
tau=0.005
max_t=200
gamma=0.9
memory_size=2000
warmup=500
batchsize=32
env=gym.make('Pendulum-v0')
device="cpu"
env=env.unwrapped
n_action=1
n_state=env.observation_space.shape[0]
max_action = float(env.action_space.high[0])

class DDPG():
    def __init__(self):
        self.actor=Actor(n_state,n_action,max_action).to(device)
        self.target_actor=Actor(n_state,n_action,max_action).to(device)
        self.critic=Critic(n_state,n_action).to(device)
        self.target_critic=Critic(n_state,n_action).to(device)
        self.memory=replay_memory(memory_size)
        self.Aoptimizer=th.optim.Adam(self.actor.parameters(),lr=lr)
        self.Coptimizer=th.optim.Adam(self.critic.parameters(),lr=lr)
    
    def actor_learn(self,batch):
        b_s=th.FloatTensor(batch[:,0].tolist()).to(device)

        loss=-self.critic(b_s,self.actor(b_s)).mean()
        self.Aoptimizer.zero_grad()
        loss.backward()
        self.Aoptimizer.step()

    def critic_learn(self,batch):
        b_s=th.FloatTensor(batch[:,0].tolist()).to(device)
        b_r=th.FloatTensor(batch[:,1].tolist()).to(device)
        b_a=th.FloatTensor(batch[:,2].tolist()).to(device)
        b_s_=th.FloatTensor(batch[:,3].tolist()).to(device)
        b_d=th.FloatTensor(batch[:,4].tolist()).to(device)

        eval_q=self.critic(b_s,b_a)
        
        next_action=th.nn.functional.softmax(self.target_actor(b_s_),dim=1)

        index=th.argmax(next_action,dim=1).unsqueeze(1)
        next_action=th.zeros_like(next_action).scatter_(1,index,1).to(device)

        target_q=th.zeros_like(eval_q).to(device)

        for i in range(b_d.shape[0]):

            target_q[i]=(1-b_d[i,0])*gamma*self.target_critic(b_s_,next_action)[i]+b_r[i]
        td_error=eval_q-target_q.detach()
        loss=(td_error**2).mean()
        self.Coptimizer.zero_grad()
        loss.backward()
        self.Coptimizer.step()


    def soft_update(self):
        for param,target_param in zip(self.actor.parameters(),self.target_actor.parameters()):
            target_param.data.copy_(tau*param.data+(1-tau)*target_param.data)
        for param,target_param in zip(self.critic.parameters(),self.target_critic.parameters()):
            target_param.data.copy_(tau*param.data+(1-tau)*target_param.data)

ddpg=DDPG()
for episode in range(2000):
    s=env.reset()
    t=0
    total_reward=0
    while(t<max_t):
        a=ddpg.actor(th.FloatTensor(s))
        a=a.data

        a+= th.FloatTensor(np.random.normal(0, 0.1, size=env.action_space.shape[0]).clip(env.action_space.low, env.action_space.high))
        s_,r,done,_=env.step(a)
        total_reward+=r
        transition=[s,[r],[a],s_,[done]]
        ddpg.memory.store(transition)
        if done:
            break
        s=s_
        if(ddpg.memory.size()<warmup):
            continue
        batch=ddpg.memory.sample(batchsize)

        ddpg.critic_learn(batch)
        ddpg.actor_learn(batch)
        ddpg.soft_update()

        t+=1
    print("episode:"+format(episode)+",test score:"+format(total_reward))
