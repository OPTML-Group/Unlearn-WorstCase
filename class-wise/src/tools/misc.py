import numpy as np
import torch
import random

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def rename_attribute(obj, old_name, new_name):
    print(obj.__dict__.keys())
    obj.__dict__[new_name] = obj.__dict__.pop(old_name)

def override_func(inst, func, func_name):
    bound_method = func.__get__(inst, inst.__class__)
    setattr(inst, func_name, bound_method)