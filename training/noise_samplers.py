import random

import torch
from torch import nn


def pyramid_noise_like(x, discount=0.8):
    b, c, w, h = x.shape # EDIT: w and h get over-written, rename for a different variant!
    u = nn.Upsample(size=(w, h), mode='bilinear')
    noise = torch.randn_like(x)
    for i in range(10):
        r = random.random()*2+2 # Rather than always going 2x, 
        w, h = max(1, int(w/(r**i))), max(1, int(h/(r**i)))
        weight = discount**i
        noise += u(torch.randn(b, c, w, h).to(x)) * weight
        if w==1 or h==1: break # Lowest resolution is 1x1
    return noise/noise.std() # Scaled back to roughly unit variance

def annealed_pyramid_noise_like(x, timesteps, total_steps, discount=0.8):
    b, c, w, h = x.shape # EDIT: w and h get over-written, rename for a different variant!
    u = nn.Upsample(size=(w, h), mode='bilinear')
    noise = torch.randn_like(x)
    for i in range(10):
        r = random.random()*2+2 # Rather than always going 2x, 
        w, h = max(1, int(w/(r**i))), max(1, int(h/(r**i)))
        steps = timesteps.view(-1, 1, 1, 1).expand(x.shape)
        weights = (discount * steps / total_steps) ** i
        n = u(torch.randn(b, c, w, h).to(x))
        noise += n * weights
        if w==1 or h==1: break # Lowest resolution is 1x1
    return noise/noise.std() # Scaled back to roughly unit variance
