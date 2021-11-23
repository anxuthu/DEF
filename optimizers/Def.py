import torch
from torch.optim.optimizer import Optimizer, required
import torch.distributed as dist
import numpy as np


class DEF(Optimizer):

    def __init__(self, params, lr=required, momentum=0, weight_decay=0,
                 reducer=required, reducer2=required, coeff=0, period=1,
                 **kwargs):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(DEF, self).__init__(params, defaults)

        self.reducer = reducer
        self.deltas = []
        self.c_deltas = []
        self.errors = []

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['old'] = p.clone().detach()

                state['delta'] = p.clone().detach().zero_()
                state['c_delta'] = p.clone().detach().zero_()
                state['error'] = p.clone().detach().zero_()
                self.deltas.append(state['delta'])
                self.c_deltas.append(state['c_delta'])
                self.errors.append(state['error'])

                state['momentum_buffer'] = p.clone().detach().zero_()

        self.coeff = coeff
        self.period = period
        self.cur_step = 0

    def __setstate__(self, state):
        super(DEF, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        self.cur_step += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                state = self.state[p]
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                d_p = state['momentum_buffer'].mul_(momentum).add_(d_p)
                p.data.add_(d_p, alpha=-group['lr'])
                state['delta'].add_(d_p, alpha=group['lr'])

        if self.cur_step % self.period == 0:
            self.reducer.reduce(self.deltas, self.c_deltas, self.errors)

            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]

                    state['old'].add_(state['c_delta'], alpha=-1.0)
                    p.data.copy_(state['old']).add_(state['error'], alpha=-self.coeff)
                    state['delta'].copy_(state['error'])

        return loss

    @torch.no_grad()
    def swap(self): # only at evaluation
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                if not 'flag' in state:
                    state['flag'] = True
                else:
                    state['flag'] = not state['flag']

                if state['flag']:
                    error = state['error'].clone()
                    dist.all_reduce(error)
                    error.div_(dist.get_world_size())
                    state['tmp'] = p.data.clone()
                    p.data.copy_(state['old'] - error)
                else:
                    p.data.copy_(state['tmp'])

    #@torch.no_grad()
    #def swap(self):
    #    for group in self.param_groups:
    #        for p in group['params']:
    #            state = self.state[p]
    #            tmp = p.data.clone().detach()
    #            p.data.copy_(state['old'])
    #            state['old'].copy_(tmp)
