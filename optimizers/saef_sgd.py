import torch
from torch.optim.optimizer import Optimizer, required
import torch.distributed as dist
import numpy as np

from .reducer import RankKReducer as Reducer


class SAEFSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v_{t+1} = \mu * v_{t} + g_{t+1} \\
                  p_{t+1} = p_{t} - lr * v_{t+1}

        where p, g, v and :math:`\mu` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v_{t+1} = \mu * v_{t} + lr * g_{t+1} \\
             p_{t+1} = p_{t} - v_{t+1}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, weight_decay=0, period2=None,
                 # power sgd
                 seed=0, device=0, prank=0, **kwargs):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(SAEFSGD, self).__init__(params, defaults)

        self.period2 = period2
        self.cur_step = 0

        self.reducer = Reducer(random_seed=seed, device=device, rank=prank)
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

    def __setstate__(self, state):
        super(SAEFSGD, self).__setstate__(state)

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
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                state = self.state[p]
                state['delta'].copy_(d_p + self.last_lr / group['lr'] * state['error'])

        c_bits = self.reducer.reduce(self.deltas, self.deltas, self.errors)
        #bits = np.sum([8 * t.nelement() * t.element_size() for t in self.deltas])
        #print('compression ratio', bits / c_bits, flush=True)
        #import sys; sys.exit()

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                p.data.copy_(state['old']).add_(state['delta'], alpha=-group['lr'])

                state['old'].copy_(p.data)
                p.data.add_(state['error'], alpha=-group['lr'])

        self.last_lr = self.param_groups[0]['lr']

        self.cur_step += 1
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
