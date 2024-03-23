import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .impl import iterative_unlearn, w_iterative_unlearn


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction="sum") * (self.T**2) / y_s.shape[0]
        return loss


def adjust_learning_rate(
    epoch, lr_decay_epochs, lr_decay_rate, sgda_learning_rate, optimizer
):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    new_lr = sgda_learning_rate
    if steps > 0:
        new_lr = sgda_learning_rate * (lr_decay_rate**steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
    return new_lr


def param_dist(model, swa_model, p):
    # This is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
    dist = 0.0
    for p1, p2 in zip(model.parameters(), swa_model.parameters()):
        dist += torch.norm(p1 - p2, p="fro")
    return p * dist


@iterative_unlearn
def SCRUB(data_loaders, model_s, model_t, eval_model_s, eval_model_t, swa_model,
          criterion_cls, criterion_div, criterion_kd, 
          optimizer, epoch, args, mask=None):
    
    forget_loader = data_loaders["forget"]
    remain_loader = data_loaders["retain"]

    lr = adjust_learning_rate(
        epoch, args.lr_decay_epochs, args.lr_decay_rate, args.theta_lr, optimizer
    )
    if epoch > args.m_steps:
        for batch in remain_loader:
            batch = [data.cuda() for data in batch]
            input = batch[:-1]
            label = batch[-1]
            
            logit_s = eval_model_s(*input)
            with torch.no_grad():
                logit_t = eval_model_t(*input)
            optimizer.zero_grad()
            loss_cls = criterion_cls(logit_s, label)
            loss_div = criterion_div(logit_s, logit_t)
            loss = args.scrub_gamma * loss_cls + args.scrub_alpha * loss_div
            loss = loss + param_dist(
                model_s, swa_model, args.smoothing
            )
            
            loss.backward()
            optimizer.step()

    elif epoch <= args.m_steps:
        for batch in forget_loader:
            batch = [data.cuda() for data in batch]
            input = batch[:-1]
            label = batch[-1]
            logit_s = eval_model_s(*input)
            with torch.no_grad():
                logit_t = eval_model_t(*input)
            optimizer.zero_grad()
            loss_div = criterion_div(logit_s, logit_t)
            loss = -loss_div + param_dist(
                model_s, swa_model, args.smoothing
            )
            
            loss.backward()
            optimizer.step()
            
    return


@w_iterative_unlearn
def w_SCRUB(data_loaders, model_s, model_t, eval_model_s, eval_model_t, swa_model,
          criterion_cls, criterion_div, criterion_kd, 
          optimizer, epoch, args, mask=None):
    
    forget_loader = data_loaders["forget"]
    remain_loader = data_loaders["retain"]

    lr = adjust_learning_rate(
        epoch, args.lr_decay_epochs, args.lr_decay_rate, args.theta_lr, optimizer
    )
    if epoch > args.m_steps:
        for batch in remain_loader:
            batch = [data.cuda() for data in batch]
            input = batch[:-2]
            label = batch[-2]
            
            logit_s = eval_model_s(*input)
            with torch.no_grad():
                logit_t = eval_model_t(*input)
            optimizer.zero_grad()
            loss_cls = criterion_cls(logit_s, label)
            loss_div = criterion_div(logit_s, logit_t)
            loss = args.scrub_gamma * loss_cls + args.scrub_alpha * loss_div
            loss = loss + param_dist(
                model_s, swa_model, args.smoothing
            )
            
            loss.backward()
            optimizer.step()

    elif epoch <= args.m_steps:
        for batch in forget_loader:
            batch = [data.cuda() for data in batch]
            
            input = batch[:-2]
            label = batch[-2]
            logit_s = eval_model_s(*input)
            with torch.no_grad():
                logit_t = eval_model_t(*input)
            optimizer.zero_grad()
            loss_div = criterion_div(logit_s, logit_t)
            loss = -loss_div + param_dist(
                model_s, swa_model, args.smoothing
            )
            
            loss.backward()
            optimizer.step()
            
    return