import torch
from torch.optim.optimizer import Optimizer, required
import torch.distributed as dist


class LSGD(Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, period=1, **kwargs):
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
        super(LSGD, self).__init__(params, defaults)

        self.period = period
        self.cur_step = 0

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['old'] = p.data.clone().detach()

    def __setstate__(self, state):
        super(LSGD, self).__setstate__(state)
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
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                p.data.add_(d_p, alpha=-group['lr'])

        if self.cur_step % self.period == 0:
            for group in self.param_groups:
                momentum = group['momentum']
                dampening = group['dampening']
                nesterov = group['nesterov']

                for p in group['params']:
                    state = self.state[p]
                    d_p = (state['old'] - p.data) / group['lr']
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

        return loss
