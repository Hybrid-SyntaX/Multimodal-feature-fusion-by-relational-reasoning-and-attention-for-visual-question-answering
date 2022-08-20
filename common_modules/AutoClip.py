import numpy as np
import torch

import warnings


def compute_weights_norm(model, norm_type=2):
    total_norm = 0
    for p in model.parameters():
        if p is not None:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    return total_norm

def compute_grad_norm(model, norm_type=2):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            #p.data=torch.zeros_like(p.data)
            #torch.fill_(p.data, float('1e999'))
            #with torch.no_grad():
                #torch.clamp_(p.data, 1e-16,1e16)
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    return total_norm

class AutoClip:
    def __init__(self, percentile):
        self.grad_history = []
        self.percentile = percentile




    def __call__(self, model):
        grad_norm = compute_grad_norm(model,norm_type = 2)
        if not np.isinf(grad_norm) and not np.isnan(grad_norm):
            self.grad_history.append(grad_norm)
        else:
            warnings.warn(f"grad_norm value is {grad_norm}")

        #print(self.grad_history)
        if len(self.grad_history) >0:
            clip_value = np.percentile(self.grad_history, self.percentile)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        else:
            clip_value = 0


        return grad_norm, clip_value
