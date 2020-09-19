import gym
import torch as th
import numpy as np 

batch_size=50
lr=0.001
episilon=0.5
replay_memory_size=10000
gamma=0.9
target_update_iter=100
env=gym.make('CartPole-v0')
env=env.unwrapped
n_action=env.action_space.n
n_state=env.observation_space.shape[0]
hidden=32

class net(th.nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc1=th.nn.Linear(n_state,hidden)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out=th.nn.Linear(hidden,n_action)
        self.out.weight.data.normal_(0, 0.1)
    

    def forward(self,x):
        x=self.fc1(x)
        x=th.nn.functional.relu(x)
        out=self.out(x)
        return out

class replay_memory():
    def __init__(self):
        self.memory_size=replay_memory_size
        self.memory=np.array([])
        self.cur=0
        self.new=0
    def size(self):
        return self.memory.shape[0]
#[s,a,r,s_,done]
    def store_transition(self,trans):
        if(self.memory.shape[0]<self.memory_size):
            if self.new==0:
                self.memory=np.array(trans)
                self.new=1
            elif self.memory.shape[0]>0:
                self.memory=np.vstack((self.memory,trans))

        else:
            self.memory[self.cur,:]=trans
            self.cur=(self.cur+1)%self.memory_size
    
    def sample(self):
        if self.memory.shape[0]<batch_size:
            return -1
        sam=np.random.choice(self.memory.shape[0],batch_size)
        return self.memory[sam]
    
class DQN(object):
    def __init__(self):
        self.eval_q_net,self.target_q_net=net(),net()
        self.replay_mem=replay_memory()
        self.iter_num=0
        self.optimizer=th.optim.Adam(self.eval_q_net.parameters(),lr=lr)
        self.loss=th.nn.MSELoss()
    def choose_action(self,qs):
        if np.random.uniform()<episilon:
            return th.argmax(qs).tolist()
        else:
            return np.random.randint(0,n_action)
    def greedy_action(self,qs):
        return th.argmax(qs).numpy()
    def learn(self):
        if(self.iter_num%target_update_iter==0):
            self.target_q_net.load_state_dict(self.eval_q_net.state_dict())
        self.iter_num+=1

        batch=self.replay_mem.sample()
        b_s=th.FloatTensor(batch[:,0].tolist())
        b_a=th.LongTensor(batch[:,1].astype(int).tolist())
        b_r=th.FloatTensor(batch[:,2].tolist())
        b_s_=th.FloatTensor(batch[:,3].tolist())
        b_d=th.FloatTensor(batch[:,4].tolist())
        q_target=th.zeros((batch_size,1))
        q_eval=self.eval_q_net(b_s)
        q_eval=th.gather(q_eval,dim=1,index=th.unsqueeze(b_a,1))
        q_next=self.target_q_net(b_s_).detach()
        for i in range(b_d.shape[0]):
            if(int(b_d[i].tolist()[0])==0):
                q_target[i]=b_r[i]+gamma*th.unsqueeze(th.max(q_next[i],0)[0],0)
            else:
                q_target[i]=b_r[i]
        td_error=self.loss(q_eval,q_target)

        self.optimizer.zero_grad()
        td_error.backward()
        self.optimizer.step()

dqn=DQN()

for episode in range(10000):
    s=env.reset()
    t=0
    r=0.0
    while(t<300):
        t+=1
        qs=dqn.eval_q_net(th.FloatTensor(s))
        a=dqn.choose_action(qs)
        transition=[s.tolist(),a,[r],s_.tolist(),[done]]
        dqn.replay_mem.store_transition(transition)
        s=s_
        if dqn.replay_mem.size()>batch_size:
            dqn.learn()
        if done:
            break
    if episode%100==0: #test
        total_reward=0.0
        for i in range(10):
            t_s=env.reset()
            t_r=0.0
            tr=0.0
            time=0
            while(time<300):
                time+=1
                t_qs=dqn.eval_q_net(th.FloatTensor(t_s))
                t_a=dqn.greedy_action(t_qs)
                ts_,tr,tdone,_=env.step(t_a)
                t_r+=tr
                if tdone:
                    break
                t_s=ts_
            total_reward+=t_r
        print("episode:"+format(episode)+",test score:"+format(total_reward/10))




