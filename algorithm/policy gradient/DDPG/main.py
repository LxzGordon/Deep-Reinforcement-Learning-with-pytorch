import gym
import torch as th
import numpy as np
from model import Critic,Actor
from experience_replay import replay_memory


lr=0.001
tau=0.05
max_t=200
gamma=0.9
memory_size=2000
warmup=500
batchsize=32
env=gym.make('Pendulum-v0')
device="cuda"
env=env.unwrapped
n_action=1
n_state=env.observation_space.shape[0]
max_action = float(env.action_space.high[0])
#print(max_action)
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
        action=self.actor(b_s)
        #print(action)
        loss=-(self.critic(b_s,action).mean())
        self.Aoptimizer.zero_grad()
        loss.backward()
        self.Aoptimizer.step()
    
    def critic_learn(self,batch):
        b_s=th.FloatTensor(batch[:,0].tolist()).to(device)
        b_r=th.FloatTensor(batch[:,1].tolist()).to(device)
        b_a=th.FloatTensor(batch[:,2].tolist()).to(device)
        b_s_=th.FloatTensor(batch[:,3].tolist()).to(device)
        b_d=th.FloatTensor(batch[:,4].tolist()).to(device)

        next_action=self.target_actor(b_s_)
        #print(next_action)
        target_q=self.target_critic(b_s_,next_action)
        for i in range(b_d.shape[0]):
            if b_d[i]:
                target_q[i]=b_r[i]
            else:
                target_q[i]=b_r[i]+gamma*target_q[i]
        eval_q=self.critic(b_s,b_a)

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

def main():
    var=3
    for episode in range(2000):
        s=env.reset()
        total_reward=0
        Normal=th.distributions.normal.Normal(th.FloatTensor([0]),th.FloatTensor([var]))
        t=0
        while t<max_t:
            noise=th.clamp(Normal.sample(),env.action_space.low[0], env.action_space.high[0]).to(device)
            a=ddpg.actor(th.FloatTensor(s).to(device))+noise
            a=th.clamp(a,env.action_space.low[0], env.action_space.high[0]).to(device)
            
            s_,r,done,_=env.step(a.tolist())
            total_reward+=r
            transition=[s,[r],[a],s_,[done]]
            ddpg.memory.store(transition)
            #print(done)
            if done:
                break
            s=s_
            if(ddpg.memory.size()<warmup):
                #print(ddpg.memory.size())
                continue
            var*=0.9995
            batch=ddpg.memory.sample(batchsize)
            ddpg.critic_learn(batch)
            ddpg.actor_learn(batch)

            ddpg.soft_update()
            t+=1
        print("episode:"+format(episode)+",test score:"+format(total_reward)+',variance:',var)

if __name__=='__main__':
    main()
