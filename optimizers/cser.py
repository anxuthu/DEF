import torch
from torch.optim.optimizer import Optimizer, required
import torch.distributed as dist
import numpy as np


class CSER(Optimizer):

    def __init__(self, params, lr=required, momentum=0, weight_decay=0,
                 reducer=required, reducer2=required, period2=0,
                 **kwargs):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(CSER, self).__init__(params, defaults)

        self.reducer = reducer
        self.deltas = []
        self.c_deltas = []
        self.r_deltas = []
        self.reducer2 = reducer2
        self.errors = []
        self.c_errors = []
        self.r_errors = []

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['old'] = p.clone().detach()

                state['delta'] = p.clone().detach().zero_()
                state['c_delta'] = p.clone().detach().zero_()
                state['r_delta'] = p.clone().detach().zero_()
                self.deltas.append(state['delta'])
                self.c_deltas.append(state['c_delta'])
                self.r_deltas.append(state['r_delta'])

                state['error'] = p.clone().detach().zero_()
                state['c_error'] = p.clone().detach().zero_()
                state['r_error'] = p.clone().detach().zero_()
                self.errors.append(state['error'])
                self.c_errors.append(state['c_error'])
                self.r_errors.append(state['r_error'])

                state['momentum_buffer'] = p.clone().detach().zero_()

        self.period2 = period2
        self.cur_step = 0

    def __setstate__(self, state):
        super(CSER, self).__setstate__(state)

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
                state['delta'].copy_(d_p * group['lr'])

        self.reducer.reduce(self.deltas, self.c_deltas, self.r_deltas) # psync

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                delta = state['c_delta'] + state['r_delta'] # psync
                p.data.add_(delta, alpha=-1.0)
                state['error'].add_(state['r_delta'], alpha=-1.0)

        if self.period2 and self.cur_step % self.period2 == 0:
            self.reducer2.reduce(self.errors, self.c_errors, self.r_errors) # psync

            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    error = state['c_error'] + state['r_error'] # psync
                    p.data.sub_(state['error']).add_(error)
                    state['error'].copy_(state['r_error'])

        return loss
