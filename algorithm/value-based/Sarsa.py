import numpy as np
import gym
import torch as th

env=gym.make('CartPole-v0')
gamma=0.9
episilon=0.9
lr=0.001
target_update_iter=100
log_internval=10
env=gym.make('CartPole-v0')
device="cuda"
env=env.unwrapped
n_action=env.action_space.n
n_state=env.observation_space.shape[0]
hidden=256

class net(th.nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc1=th.nn.Linear(n_state,hidden)
        self.out=th.nn.Linear(hidden,n_action)
    

    def forward(self,x):
        x=self.fc1(x)
        x=th.nn.functional.relu(x)
        out=self.out(x)
        return out

class Sarsa():
    def __init__(self):
        self.net,self.target_net=net(),net()
        self.iter_num=0
        self.optimizer=th.optim.Adam(self.net.parameters(),lr=lr)

    def learn(self,s,a,s_,r,done):
        eval_q=self.net(th.Tensor(s))[a]
        target_q=self.target_net(th.FloatTensor(s_))
        target_a=self.choose_action(target_q)
        target_q=target_q[target_a]
        if not done:
            y=gamma*target_q+r
        else:
            y=r
        loss=(y-eval_q)**2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.iter_num+=1
        if self.iter_num%10==0:
            self.target_net.load_state_dict(self.net.state_dict())
        return target_a

    def greedy_action(self,qs):
        return th.argmax(qs)

    def random_action(self):
        return np.random.randint(0,n_action)

    def choose_action(self,qs):
        if np.random.rand()>episilon:
            return self.random_action()
        else:
            return self.greedy_action(qs).tolist()

sarsa=Sarsa()
for episode in range(10000):
    s=env.reset()
    t=0
    r=0.0
    qs=sarsa.net(th.Tensor(s))
    a=sarsa.choose_action(qs)
    while(t<300):
        t+=1
        #print(a)
        s_,r,done,_=env.step(a)
        a=sarsa.learn(s,a,s_,r,done)
        s=s_
        if done:
            break
    if episode%log_internval==0: #test
        total_reward=0.0
        for i in range(10):
            t_s=env.reset()
            t_r=0.0
            tr=0.0
            time=0
            while(time<300):
                time+=1
                qs=sarsa.net(th.Tensor(t_s))
                a=sarsa.greedy_action(qs)
                ts_,tr,tdone,_=env.step(a.tolist())
                t_r+=tr
                if tdone:
                    break
                t_s=ts_
            total_reward+=t_r
        print("episode:"+format(episode)+",test score:"+format(total_reward/10))



        


