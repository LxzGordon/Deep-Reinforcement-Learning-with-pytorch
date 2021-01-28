import torch as th


class SharedAdam(th.optim.Adam):
    #update parameters in parallel
    #Code from https://github.com/MorvanZhou/pytorch-A3C/blob/master/shared_adam.py
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] =0
                state['exp_avg'] = th.zeros_like(p.data)
                state['exp_avg_sq'] = th.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

def update_global(optA,optC,gnet,lnet,bs,ba,br,bs_,bd):
    gamma=0.9
    bs=th.Tensor(bs)
    bs_=th.Tensor(bs_)
    ba=th.LongTensor(ba)
    br=th.Tensor(br)
    #critic
    v_target,_=lnet(bs_)
    v_target=v_target*gamma+br
    v_eval,prob=lnet(bs)
    for i in range(br.shape[0]):
        if bd[i]:
            v_target=br[i]
    td_error=v_target.detach()-v_eval
    loss_c=(td_error**2).mean()
    optC.zero_grad()
    loss_c.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    optC.step()
    #actor
    prob=th.gather(prob,1,ba)
    log_prob=th.log(prob)
    loss_a=(-td_error.detach()*log_prob).mean()

    optA.zero_grad()
    loss_a.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    optA.step()

    lnet.load_state_dict(gnet.state_dict())


