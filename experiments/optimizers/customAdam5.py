import json
import torch
import torch.nn as nn
import math

class customAdam5(torch.optim.Optimizer):
    torch.autograd.set_detect_anomaly(True)
    torch.set_grad_enabled(True)
    def __init__(self, params, lr = 1e-3, betas = (0.99,0.999), eps = 1e-3):
        defaults = dict(lr = lr, betas = betas, eps = eps)
        super(customAdam5,self).__init__(params,defaults)

    def __setstate__(self, state) -> None:
        return super(customAdam5).__setstate__(state)

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
                
                beta1_d = torch.zeros_like(state['exp_avg'])

                # for i in range(len(grad)):
                #     for j in range(len(grad[i])):
                #         if abs(grad[i][j]/state['exp_avg'][i][j]) < 1e-1:
                #             beta1_d[i][j] = beta1**(abs(state['exp_avg'][i][j]/grad[i][j]))
                #         else:
                #             beta1_d[i][j] = beta1
                condition = torch.abs(grad / state['exp_avg']) < 1e-1
                beta1_d = torch.where(condition, beta1**(torch.abs(state['exp_avg'] / (grad + group['eps']))), beta1)

                beta2_d = beta2

    

                state['exp_avg'] = beta1_d * state['exp_avg'] + (1 - beta1_d) * grad
                state['exp_avg_sq'] = beta2_d * state['exp_avg_sq'] + (1 - beta2_d) * (grad**2)

                bias_correction1 = 1 - beta1**state['step']
                bias_correction2 = 1 - beta2**state['step']
           
                p.data.addcdiv_(state['exp_avg'], (state['exp_avg_sq'] / (bias_correction2 + group['eps'])).sqrt() + group['eps'], value=-group['lr'] / (bias_correction1 +  group['eps']))

        return loss
