import torch as th 
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
import gym
from utils import update_global

max_lstep=200
update_interval=10

class Net(nn.Module):
    def __init__(self,n_state,n_action):
        super(Net,self).__init__()
        self.n_state=n_state
        self.n_action=n_action
        self.v=nn.Sequential(
                              nn.Linear(n_state,256),
                              nn.Linear(256,1))
        self.policy=nn.Sequential(
                              nn.Linear(n_state,256),
                              nn.Linear(256,n_action))
    def forward(self,x):
        value=self.v(x)
        prob=F.softmax(self.policy(x))
        #print(prob)
        return value,prob

class Worker(mp.Process):
    def __init__(self,gnet,optA,optC,name):
        super(Worker,self).__init__()
        self.name='Worker '+name
        self.optA=optA
        self.optC=optC
        self.env=gym.make('CartPole-v0')
        self.lnet=Net(self.env.observation_space.shape[0],self.env.action_space.n)
        self.gnet=gnet
        self.queue=[]
        self.max_episode=1000
        self.cur_episode=0
    def choose_action(self,s):
        _,prob=self.lnet(th.Tensor(s))
        dist=th.distributions.categorical.Categorical(prob)
        a=dist.sample().tolist()
        return a

    def run(self):
        buffer_a,buffer_s,buffer_r,buffer_s_,buffer_d=[],[],[],[],[]
        while self.cur_episode<self.max_episode:
            s=self.env.reset()
            total_reward=0
            total_step=0
            for _ in range(max_lstep):
                a=self.choose_action(s)
                s_,r,done,_=self.env.step(a)
                total_reward+=r
                buffer_a.append([a])
                buffer_s.append(s)
                buffer_r.append([r])
                buffer_s_.append(s_)
                buffer_d.append(done)
                #print(done)
                s=s_
                total_step+=1
                if total_step%update_interval==0 or done:
                    #opt,gnet,lnet,bs,ba,br,bs_
                    update_global(self.optA,self.optC,self.gnet,self.lnet,buffer_s,buffer_a,buffer_r,buffer_s_,buffer_d)
                    buffer_a,buffer_s,buffer_r,buffer_s_,buffer_d=[],[],[],[],[]
                    if done:
                        #print('done')
                        self.queue.append(total_reward)
                        if len(self.queue)==11:
                            del self.queue[0]
                        #print(self.queue)
                        break

            self.cur_episode+=1
            if self.name=='Worker 0':
                if self.cur_episode%10==0:
                    #print(self.cur_episode,th.mean(th.Tensor(self.queue)))
                    print(self.cur_episode,self.queue)


