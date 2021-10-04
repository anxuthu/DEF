import torch
from torch.optim.optimizer import Optimizer, required
import torch.distributed as dist


class LSGD2(Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, period=1, period2=1, coeff=1, **kwargs):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(LSGD2, self).__init__(params, defaults)

        assert period % period == 0
        self.period = period
        self.period2 = period2
        self.cur_step = 0
        self.coeff = 1

    def __setstate__(self, state):
        super(LSGD2, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

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

            for p in group['params']:
                state = self.state[p]

                if not 'old' in state:
                    state['old'] = p.data.clone().detach()

                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                p.data.add_(d_p, alpha=-group['lr'])

                if not 'delta' in state:
                    state['delta'] = p.data.clone().detach().zero_()
                state['delta'].add_(d_p)

        if self.cur_step % self.period2 == 0:
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    p.data[:] = state['old'] - self.coeff * group['lr'] * state['delta']

        if self.cur_step % self.period == 0:
            for group in self.param_groups:
                momentum = group['momentum']
                dampening = group['dampening']
                nesterov = group['nesterov']

                for p in group['params']:
                    state = self.state[p]
                    d_p = state['delta']
                    dist.all_reduce(d_p)
                    d_p /= dist.get_world_size()

                    if momentum != 0:
                        if 'momentum_buffer' not in state:
                            buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    state['old'].add_(d_p, alpha=-group['lr'])
                    p.data.copy_(state['old'])
                    state['delta'].zero_()

        return loss
