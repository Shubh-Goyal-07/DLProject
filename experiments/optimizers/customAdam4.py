import json
import torch
import torch.nn as nn
import math

class customAdam4(torch.optim.Optimizer):
    torch.autograd.set_detect_anomaly(True)
    def __init__(self, params, lr = 1e-3, betas = (0.99,0.999,0.9999), eps = 1e-3):
        defaults = dict(lr = lr, betas = betas, eps = eps)
        super(customAdam4,self).__init__(params,defaults)

    def __setstate__(self, state) -> None:
        return super(customAdam4).__setstate__(state)

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
                    state['exp_avg_cube'] = torch.zeros_like(p.data)
                    # init for third moment
                    # state['exp_avg_third'] = torch.zeros_like(p.data)
                    
                beta1, beta2, beta3 = group['betas']
                state['step'] += 1
                # beta1,beta2 = min(beta1*math.exp(-state['step']), 0.99) ,min(beta2*math.exp(-state['step']), 0.99)

                state['grad'] = grad
                state['exp_avg'] = beta1 * state['exp_avg'] + (1 - beta1) * grad
                state['exp_avg_cube'] = beta2 * state['exp_avg_cube'] + (1 - beta2) * grad**3
                # Thrid Moment
                # state['exp_avg_third'] = beta3 * state['exp_avg_third'] + (1 - beta3) * grad**3

                # print("s_cap",state['exp_avg'])
                # print("r_cap",state['exp_avg_cube'])

                bias_correction1 = 1 - beta1**state['step']
                bias_correction2 = 1 - beta2**state['step']
                # bias_correction3 = 1 - beta3**state['step']

                # Utility expression for degree balance
                s_cap = torch.pow(state['exp_avg']/bias_correction1,0.75)
                r_cap = torch.pow(state['exp_avg_cube']/bias_correction2,0.25)
                # t_cap = (state['exp_avg_third']/(bias_correction3))**(1/3)

                s_cap[s_cap != s_cap] = 1e-15
                r_cap[r_cap != r_cap] = 1e-15
                
                # t_cap[t_cap != t_cap] = 1e-9

                # s_cap = torch.max(s_cap,torch.nan)
                # r_cap = torch.max(r_cap,torch.nan)
                # t_cap = torch.max(t_cap,torch.nan)

                # print("s_cap:", s_cap/bias_correction1)
                # print("r_cap:", r_cap/bias_correction2)
                # print("t_cap:", t_cap)
                # print(r_cap)
                # Third Moment final expression
                p.data.addcdiv_(s_cap, r_cap + group['eps'], value=-group['lr'])

                # else:
                #     state['step'] += 1
                #     state['exp_avg'] = beta1 * state['exp_avg'] + (1 - beta1) * grad
                #     state['exp_avg_sq'] = beta2 * state['exp_avg_sq'] + (1 - beta2) * grad**2

                #     bias_correction1 = 1 - beta1**state['step']
                #     bias_correction2 = 1 - beta2**state['step']

                #     p.data.addcdiv_(state['exp_avg'], (state['exp_avg_sq'] / bias_correction2).sqrt() + group['eps'], value=-group['lr'] / bias_correction1)
                #     p.data.addcdiv_(state['exp_avg'], (state['exp_avg_sq'] / bias_correction2).sqrt() + group['eps'], value=-group['lr'] / bias_correction1)

        return loss
