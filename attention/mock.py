import numpy as np 
from matplotlib import pyplot as plt 
import torch 
from debug import debugs, debug, debugt

a = torch.tensor([
    [1,2],
    [3,4],
])

b = torch.tensor([
    [1,0],
    [0,1],
])

a = torch.stack([a for i in range(1,4)])
b = torch.stack([b*i for i in range(1,4)])

debug(a)
debug(b)
debug(a@b)

