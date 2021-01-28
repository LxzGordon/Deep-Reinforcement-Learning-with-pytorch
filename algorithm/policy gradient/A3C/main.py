import torch as th
from utils import SharedAdam
from model import Worker,Net
import gym
import torch.multiprocessing as mp

env=gym.make('CartPole-v0')
n_action=env.action_space.n
n_state=env.observation_space.shape[0]

global_net=Net(n_state,n_action)
global_net.share_memory()
optA=SharedAdam(global_net.policy.parameters(), lr=1e-4, betas=(0.92, 0.999))
optC=SharedAdam(global_net.v.parameters(), lr=1e-4, betas=(0.92, 0.999))
workers=[Worker(global_net,optA,optC,str(i)) for i in range(8)]
[w.start() for w in workers]

[w.join() for w in workers]
