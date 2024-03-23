import torch
from torch.nn import functional as F
import numpy as np
import torchmetrics


def warmup_lr(optimizer, current_epoch, current_step, steps_per_epoch, warmup_epoch, base_lr):
    overall_steps = warmup_epoch * steps_per_epoch
    current_steps = current_epoch * steps_per_epoch + current_step
    lr = base_lr * current_steps/overall_steps
    for p in optimizer.param_groups:
        p['lr']=lr



class BlurPoolConv2d(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)
    

def apply_blurpool(mod: torch.nn.Module):
    for (name, child) in mod.named_children():
        if isinstance(child, torch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
            setattr(mod, name, BlurPoolConv2d(child))
        else: apply_blurpool(child)


class MeanScalarMetric(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state('sum', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, sample):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count