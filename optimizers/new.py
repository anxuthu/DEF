import torch
from torch.optim.optimizer import Optimizer, required
import torch.distributed as dist
import numpy as np


class New(Optimizer):

    def __init__(self, params, lr=required, momentum=0, weight_decay=0,
                 reducer=required, coeff=0, **kwargs):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(New, self).__init__(params, defaults)

        self.reducer = reducer
        self.deltas = []
        self.errors = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['old'] = p.clone().detach()
                state['delta'] = p.clone().detach().zero_()
                state['error'] = p.clone().detach().zero_()
                self.deltas.append(state['delta'])
                self.errors.append(state['error'])

        self.last_lr = 1
        self.coeff = coeff

    def __setstate__(self, state):
        super(New, self).__setstate__(state)

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

        # optimizer update
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                d_p = p.grad.data
                state = self.state[p]

                """ pre-reduce """
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                state['delta'].copy_(d_p + self.last_lr / group['lr'] * state['error'])

        c_bits = self.reducer.reduce(self.deltas, self.deltas, self.errors)

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                state = self.state[p]
                delta = state['delta']

                """ post-reduce """
                #if weight_decay != 0:
                #    delta = delta.add(p.data, alpha=weight_decay)
                #if momentum != 0:
                #    if 'momentum_buffer' not in state:
                #        buf = state['momentum_buffer'] = torch.clone(delta).detach()
                #    else:
                #        buf = state['momentum_buffer']
                #        buf.mul_(momentum).add_(delta)
                #    delta = buf

                p.data.copy_(state['old']).add_(delta, alpha=-group['lr'])

                state['old'].copy_(p.data)
                p.data.add_(state['error'], alpha=-group['lr'] * self.coeff)

        self.last_lr = self.param_groups[0]['lr']

        return loss

    @torch.no_grad()
    def swap(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                tmp = p.data.clone().detach()
                p.data.copy_(state['old'])
                state['old'].copy_(tmp)
