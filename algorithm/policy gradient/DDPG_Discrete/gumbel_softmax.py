import torch as th
import numpy as np
device='cpu'
def gumbel_sample(shape,eps=1e-10):
    seed=th.FloatTensor(shape).uniform_().to(device)
    return -th.log(-th.log(seed+eps)+eps)

def gumbel_softmax_sample(logits,temperature=1.0):
    #print(logits)
    logits=logits+gumbel_sample(logits.shape,1e-10)
    #print(logits)
    return (th.nn.functional.softmax(logits/temperature,dim=1))

def gumbel_softmax(prob,temperature=1.0,hard=False):
    #print(prob)
    logits=th.log(prob)
    y=gumbel_softmax_sample(prob,temperature)
    if hard==True:   #one hot but differenttiable
        y_onehot=onehot_action(y)
        y=(y_onehot-y).detach()+y
    return y

def onehot_action(prob):
    y=th.zeros_like(prob).to(device)
    index=th.argmax(prob,dim=1).unsqueeze(1)
    y=y.scatter(1,index,1)
    return y.to(th.long)
