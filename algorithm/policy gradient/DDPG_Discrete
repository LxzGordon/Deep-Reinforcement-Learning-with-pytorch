import gym
import torch as th
import numpy as np
from gumbel_softmax import gumbel_softmax
from model import Critic,Actor
from experience_reply import replay_memory


lr=0.001
tau=0.05
max_t=200
gamma=0.9
memory_size=2000
batchsize=32
warmup=batchsize
env=gym.make('CartPole-v0')
device="cpu"
env=env.unwrapped
n_action=env.action_space.n
n_state=env.observation_space.shape[0]

class DDPG():
    def __init__(self):
        self.actor=Actor(n_state,n_action).to(device)
        self.target_actor=Actor(n_state,n_action).to(device)
        self.critic=Critic(n_state,n_action).to(device)
        self.target_critic=Critic(n_state,n_action).to(device)
        self.memory=replay_memory(memory_size)
        self.Aoptimizer=th.optim.Adam(self.actor.parameters(),lr=lr)
        self.Coptimizer=th.optim.Adam(self.critic.parameters(),lr=lr)
    
    def choose_action(self,state,eps):
        prob=self.actor(th.FloatTensor(state).to(device))
        prob=th.nn.functional.softmax(prob,0)
        #print(prob)
        if np.random.uniform()>eps:
            action=th.argmax(prob,dim=0).tolist()
        else:
            action=np.random.randint(0,n_action)
        return action
    
    def actor_learn(self,batch):
        b_s=th.FloatTensor(batch[:,0].tolist()).to(device)
        b_r=th.FloatTensor(batch[:,1].tolist()).to(device)
        b_a=th.FloatTensor(batch[:,2].tolist()).to(device)
        
        differentiable_a=th.nn.functional.gumbel_softmax(th.log(th.nn.functional.softmax(self.actor(b_s),dim=1)),hard=True)
        #print(differentiable_a)
        #differentiable_a2=th.nn.functional.softmax(th.nn.functional.softmax(self.actor(b_s),dim=1),dim=1)
        #index=th.argmax(differentiable_a2,dim=1).unsqueeze(1)
        #oh=th.zeros_like(differentiable_a2).scatter_(1,index,1)
        #differentiable_a2=(oh-differentiable_a2).detach()+differentiable_a2

        loss=-self.critic(b_s,differentiable_a).mean()
        self.Aoptimizer.zero_grad()
        loss.backward()
        self.Aoptimizer.step()

    def critic_learn(self,batch):
        b_s=th.FloatTensor(batch[:,0].tolist()).to(device)
        b_r=th.FloatTensor(batch[:,1].tolist()).to(device)
        b_a=th.zeros(batchsize,n_action).scatter_(1,th.LongTensor(batch[:,2].tolist()),1).to(device)
        b_s_=th.FloatTensor(batch[:,3].tolist()).to(device)
        b_d=th.FloatTensor(batch[:,4].tolist()).to(device)

        eval_q=self.critic(b_s,b_a)

        next_action=th.nn.functional.softmax(self.target_actor(b_s_),dim=1)

        index=th.argmax(next_action,dim=1).unsqueeze(1)
        next_action=th.zeros_like(next_action).scatter_(1,index,1).to(device)
        print(next_action)
        target_q=th.zeros_like(eval_q).to(device)

        for i in range(b_d.shape[0]):
            target_q[i]=(1-b_d[i,0])*gamma*self.target_critic(b_s_,next_action)[i].detach()+b_r[i]
        td_error=eval_q-target_q
        loss=(td_error**2).mean()
        self.Coptimizer.zero_grad()
        loss.backward()
        self.Coptimizer.step()


    def soft_update(self):
        for param,target_param in zip(self.actor.parameters(),self.target_actor.parameters()):
            target_param.data.copy_(tau*param.data+(1-tau)*target_param.data)
        for param,target_param in zip(self.critic.parameters(),self.target_critic.parameters()):
            target_param.data.copy_(tau*param.data+(1-tau)*target_param.data)
for j in range(10):
    ddpg=DDPG()
    highest=0
    for episode in range(300):
        s=env.reset()
        t=0
        total_reward=0
        while(t<max_t):
            a=ddpg.choose_action(s,0.1)
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
        if episode%10==0:
            total_reward=0.0
            for i in range(1):
                t_s=env.reset()
                t_r=0.0
                tr=0.0
                time=0
                while(time<300):
                    time+=1
                    t_a=ddpg.choose_action(t_s,0)
                    ts_,tr,tdone,_=env.step(t_a)
                    t_r+=tr
                    if tdone:
                        break
                    t_s=ts_
                total_reward+=t_r
                if total_reward>highest:
                    highest=total_reward
                print("episode:"+format(episode)+",test score:"+format(total_reward))
    if(highest>20):
        print(format(j+1)+"th round did it")
