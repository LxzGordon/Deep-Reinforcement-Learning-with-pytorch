import gym
import torch as th
import numpy as np
from model import Critic,Actor
from experience_replay import replay_memory


lr=0.001
tau=0.005
max_t=200
gamma=0.99
c=0.5
d=2
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
class TD3():
    def __init__(self):
        self.actor=Actor(n_state,n_action,max_action).to(device)
        self.target_actor=Actor(n_state,n_action,max_action).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic1=Critic(n_state,n_action).to(device)
        self.target_critic1=Critic(n_state,n_action).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())

        self.critic2=Critic(n_state,n_action).to(device)
        self.target_critic2=Critic(n_state,n_action).to(device)
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.memory=replay_memory(memory_size)
        self.Aoptimizer=th.optim.Adam(self.actor.parameters(),lr=lr)
        self.C1optimizer=th.optim.Adam(self.critic1.parameters(),lr=lr)
        self.C2optimizer=th.optim.Adam(self.critic2.parameters(),lr=lr)

    def actor_learn(self,batch):
        b_s=th.FloatTensor(batch[:,0].tolist()).to(device)
        action=self.actor(b_s)
        loss=-(self.critic1(b_s,action).mean())
        self.Aoptimizer.zero_grad()
        loss.backward()
        self.Aoptimizer.step()
    
    def critic_learn(self,batch,policy_noise):
        b_s=th.FloatTensor(batch[:,0].tolist()).to(device)
        b_r=th.FloatTensor(batch[:,1].tolist()).to(device)
        b_a=th.FloatTensor(batch[:,2].tolist()).to(device)
        b_s_=th.FloatTensor(batch[:,3].tolist()).to(device)
        b_d=th.FloatTensor(batch[:,4].tolist()).to(device)

        dist=th.distributions.normal.Normal(th.FloatTensor([0]),th.Tensor([policy_noise]))
        next_action=self.target_actor(b_s_)+th.clamp(dist.sample(),-c,c).to(device)
        next_action=th.clamp(next_action,env.action_space.low[0],env.action_space.high[0])
        #print(th.cat((self.target_critic1(b_s_,next_action),self.target_critic2(b_s_,next_action)),dim=1))
        target_q=th.min(self.target_critic1(b_s_,next_action),self.target_critic2(b_s_,next_action))
        #print(target_q)
        for i in range(b_d.shape[0]):
            if b_d[i]:
                target_q[i]=b_r[i]
            else:
                target_q[i]=b_r[i]+gamma*target_q[i]
        #target_q=b_r+gamma*target_q
        eval_q1=self.critic1(b_s,b_a)
        td_error=eval_q1-target_q.detach()
        loss=(td_error**2).mean()
        self.C1optimizer.zero_grad()
        loss.backward()
        self.C1optimizer.step()

        eval_q2=self.critic2(b_s,b_a)
        td_error=eval_q2-target_q.detach()
        loss=(td_error**2).mean()
        self.C2optimizer.zero_grad()
        loss.backward()
        self.C2optimizer.step()

    def soft_update(self):
        for param,target_param in zip(self.actor.parameters(),self.target_actor.parameters()):
            target_param.data.copy_(tau*param.data+(1-tau)*target_param.data)
        for param,target_param in zip(self.critic1.parameters(),self.target_critic1.parameters()):
            target_param.data.copy_(tau*param.data+(1-tau)*target_param.data)
        for param,target_param in zip(self.critic2.parameters(),self.target_critic2.parameters()):
            target_param.data.copy_(tau*param.data+(1-tau)*target_param.data)
td3=TD3()

def main():
    exploration_noise=3
    policy_noise=0.2
    for episode in range(2000):
        s=env.reset()
        total_reward=0
        t=0
        while t<max_t:
            explore_N=th.distributions.normal.Normal(th.FloatTensor([0]),th.Tensor([exploration_noise]))
            #noise=th.clamp(explore_N.sample(),env.action_space.low[0], env.action_space.high[0]).to(device)
            noise=explore_N.sample()
            a=td3.actor(th.FloatTensor(s).to(device))+noise.to(device)
            #a=th.clamp(a,env.action_space.low[0], env.action_space.high[0]).to(device)
            
            s_,r,done,_=env.step(a.tolist())
            total_reward+=r
            transition=[s,[r],[a],s_,[done]]
            td3.memory.store(transition)
            if done:
                break
            s=s_
            if(td3.memory.size()<warmup):
                continue
            batch=td3.memory.sample(batchsize)
            td3.critic_learn(batch,policy_noise)
            exploration_noise*=0.995
            if not t%d:
                td3.actor_learn(batch)
                td3.soft_update()
                #policy_noise*=0.9995
            t+=1
        print("episode:"+format(episode)+",test score:"+format(total_reward))

if __name__=='__main__':
    main()
