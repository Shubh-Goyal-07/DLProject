import json
import torch
import torch.nn as nn
import math

class customAdam(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3, betas = (0.99,0.999), eps = 1e-3):
        defaults = dict(lr = lr, betas = betas, eps = eps)
        super(customAdam,self).__init__(params,defaults)

    def __setstate__(self, state) -> None:
        return super(customAdam).__setstate__(state)

    def step(self,closure = None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                beta1, beta2 = group['betas']
                state['step'] += 1
                beta1,beta2 = min(beta1*math.exp(-state['step']), 0.99) ,min(beta2*math.exp(-state['step']), 0.99)

                state['grad'] = grad
                state['exp_avg'] = beta1 * state['exp_avg'] + (1 - beta1) * grad
                state['exp_avg_sq'] = beta2 * state['exp_avg_sq'] + (1 - beta2) * grad**2

                bias_correction1 = 1 - beta1**state['step']
                bias_correction2 = 1 - beta2**state['step']

                p.data.addcdiv_(state['exp_avg'], (state['exp_avg_sq'] / bias_correction2).sqrt() + group['eps'], value=-group['lr'] / bias_correction1)

                # else:
                #     state['step'] += 1
                #     state['exp_avg'] = beta1 * state['exp_avg'] + (1 - beta1) * grad
                #     state['exp_avg_sq'] = beta2 * state['exp_avg_sq'] + (1 - beta2) * grad**2

                #     bias_correction1 = 1 - beta1**state['step']
                #     bias_correction2 = 1 - beta2**state['step']

                #     p.data.addcdiv_(state['exp_avg'], (state['exp_avg_sq'] / bias_correction2).sqrt() + group['eps'], value=-group['lr'] / bias_correction1)
                #     p.data.addcdiv_(state['exp_avg'], (state['exp_avg_sq'] / bias_correction2).sqrt() + group['eps'], value=-group['lr'] / bias_correction1)

        return loss
