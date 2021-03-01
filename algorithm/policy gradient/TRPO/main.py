import gym
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
from model import net,v

delta=0.01

alpha=0.001
max_t=200
gamma=0.9
update_interval=3
v_update_iter=10
env=gym.make('CartPole-v0')
device="cuda"
env=env.unwrapped
n_action=env.action_space.n
n_state=env.observation_space.shape[0]

class TRPO():
    def __init__(self):
        self.policy=net(n_state,n_action).to(device)
        self.old_policy=net(n_state,n_action).to(device)

        self.v=v(n_state).to(device)
        self.optim=th.optim.Adam(self.v.parameters(),lr=0.001)

    def choose_action(self,s):
        dist=th.distributions.Categorical(self.old_policy(th.Tensor(s).to(device)))
        return dist.sample().tolist()

    def flat_grad(self, grads, hessian=False):
      grad_flatten = []
      if hessian == False:
         for grad in grads:
            grad_flatten.append(grad.view(-1))
         grad_flatten = th.cat(grad_flatten)
         return grad_flatten
      elif hessian == True:
         for grad in grads:
            grad_flatten.append(grad.contiguous().view(-1))
         grad_flatten = th.cat(grad_flatten).data
         return grad_flatten

    def hessian_vector_product(self, obs, p, damping_coeff=0.1):
        #p=p.detach()
        self.policy.zero_grad()
        old_pi=self.old_policy(obs)
        pi=self.policy(obs)+1e-8
        kl=(old_pi*th.log(old_pi/pi)).sum(1)

        kl=kl.mean()

        kl_grad = th.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        kl_grad = self.flat_grad(kl_grad)

        kl_grad_p = (kl_grad * p).sum() 
        kl_hessian = th.autograd.grad(kl_grad_p, self.policy.parameters())
        kl_hessian = self.flat_grad(kl_hessian, hessian=True)
        #print(p)
        return kl_hessian + p.detach() * damping_coeff

    def cg(self, obs, b, cg_iters=10, EPS=1e-5, residual_tol=1e-10):
        # Conjugate gradient algorithm
        # (https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        x = th.zeros(b.size()).to(device)
        r = b.clone()
        p = r.clone()
        rdotr = th.dot(r,r).to(device)

        for _ in range(cg_iters):
            Ap = self.hessian_vector_product(obs, p)
            alpha = rdotr / (th.dot(p, Ap).to(device) + EPS)

            x += alpha * p
            r -= alpha * Ap
            
            new_rdotr = th.dot(r, r)
            p = r + (new_rdotr / rdotr) * p
            rdotr = new_rdotr

            if rdotr < residual_tol:
                break
        return x

    def update_params(self,policy,flat_params):
        index=0
        for p in policy.parameters():
            length=len(p.view(-1))
            params=flat_params[index:index+length]
            params=params.view(p.size())
            p.data.copy_(params)
            index+=length




    def learn(self,batch):

        for trajectory in batch:
            s=th.Tensor(trajectory[0]).to(device)
            a=th.LongTensor(trajectory[1]).to(device)
            r=th.Tensor(trajectory[2]).to(device)
            g=trajectory[3]
            
            for _ in range(v_update_iter):
                v=self.v(s)
                loss=(v-g.detach())**2
                loss=loss.sum()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            adv=g-v
            adv = (adv - adv.mean()) /(adv.std() + 1e-8)
            log_prob=th.log(th.gather(self.policy(s),1,a))
            log_prob_old=th.log(th.gather(self.old_policy(s),1,a))+1e-8
            prob=th.exp(log_prob-log_prob_old.detach())
            policy_loss=(prob*adv.detach()).mean()
            
            gra=th.autograd.grad(policy_loss,self.policy.parameters(),create_graph=True)
            gra=self.flat_grad(gra)
            x=self.cg(s,gra.data)

            H=self.hessian_vector_product(s,x)
            xHx=(H*x).sum(0)
            if xHx<0:    #unknown bug, H may not be positive definite at the beginning of training
                xHx=th.Tensor([1000]).to(device)
            if xHx>0:
                lr=th.sqrt(2*delta/xHx)
                
                self.old_policy.load_state_dict(self.policy.state_dict())
                #natural policy gradient
                params=self.policy.parameters()
                p=[]
                for i in params:
                    p.append(i.view(-1))
                params=th.cat(p)
                params+=lr*x
                self.update_params(self.policy,params)
            
trpo=TRPO()
start_time=time.time()
transition_s=[]
transition_a=[]
transition_r=[]
trajectory=[]
for episode in range(10000):
    t=0
    s=env.reset()
    total_reward=0

    while(t<300):
        a=trpo.choose_action(s)
        s_,r,done,_=env.step(a)
        total_reward+=r

        transition_s.append(s.tolist())
        transition_a.append([a])
        transition_r.append([r])

        if done:
            g=th.zeros((np.array(transition_r).shape[0]),1).to(device)
            for i in reversed(range(0,np.array(transition_r).shape[0]-1)):
                g[i]=g[i+1]*gamma+transition_r[i][0]

            trajectory.append([transition_s,transition_a,transition_r,g])
            transition_s=[]
            transition_a=[]
            transition_r=[]
            if len(trajectory)==update_interval:
                trpo.learn(np.array(trajectory))
                trajectory=[]
            break
        s=s_
        t+=1
    if episode%100==0:
        print("Episode:"+format(episode)+",total score:"+format(total_reward))
end_time=time.time()
print(end_time-start_time)
        
