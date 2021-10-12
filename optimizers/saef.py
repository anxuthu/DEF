import torch
from torch.optim.optimizer import Optimizer, required
import torch.distributed as dist
import numpy as np


class SAEF(Optimizer):

    def __init__(self, params, lr=required, momentum=0, weight_decay=0, period2=None,
                 reducer=required, **kwargs):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(SAEF, self).__init__(params, defaults)

        self.period2 = period2
        self.cur_step = 0

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

    def __setstate__(self, state):
        super(SAEF, self).__setstate__(state)

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

        # optimizer update
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                d_p = p.grad.data
                state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                if momentum != 0:
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                state['delta'].copy_(d_p * group['lr'] + state['error'])

        self.reducer.reduce(self.deltas, self.c_deltas, self.errors)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['old'].add_(state['c_delta'], alpha=-1.0)
                p.data.copy_(state['old']).add_(state['error'], alpha=-1.0)

        if self.period2 and self.cur_step % self.period2 == 0:
            for error in self.errors:
                dist.all_reduce(error)
                error.div_(dist.get_world_size())

        return loss

    @torch.no_grad()
    def swap(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                tmp = p.data.clone().detach()
                p.data.copy_(state['old'])
                state['old'].copy_(tmp)
