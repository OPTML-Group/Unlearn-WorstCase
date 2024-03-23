import sys
import time

import torch
import utils
from .impl import iterative_unlearn, w_iterative_unlearn


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)

def FT_iter(
    data_loaders, model, criterion, optimizer, epoch, args, mask=None, with_l1=False
):
    train_loader = data_loaders["retain"]

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(train_loader):
            image, target = get_x_y_from_data_dict(data, device)
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            # compute output
            output_clean = model(image)
            if epoch < args.unlearn_epochs - args.no_l1_epochs:
                current_alpha = args.alpha * (
                    1 - epoch / (args.unlearn_epochs - args.no_l1_epochs)
                )
            else:
                current_alpha = 0
            loss = criterion(output_clean, target)
            if with_l1:
                loss = loss + current_alpha * l1_regularization(model)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()
    else:
        for i, (image, target) in enumerate(train_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            image = image.cuda()
            target = target.cuda()
            if epoch < args.unlearn_steps - args.no_l1_epochs:
                current_alpha = args.alpha * (
                    1 - epoch / (args.unlearn_steps - args.no_l1_epochs)
                )
            else:
                current_alpha = 0
            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)
            if with_l1:
                loss += current_alpha * l1_regularization(model)

            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg


@iterative_unlearn
def FT(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    return FT_iter(data_loaders, model, criterion, optimizer, epoch, args, mask)


@iterative_unlearn
def FT_l1(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    return FT_iter(
        data_loaders, model, criterion, optimizer, epoch, args, mask, with_l1=True
    )


def w_FT_iter(
    train_loader, model, criterion, optimizer, epoch, args, w=None, with_l1=False
):

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()

    for i, (image, target, index) in enumerate(train_loader):
        if epoch < args.warmup:
            utils.warmup_lr(
                epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
            )

        image = image.cuda()
        target = target.cuda()
        if epoch < args.select_epochs - args.no_l1_epochs:
            current_alpha = args.alpha * (
                1 - epoch / (args.unlearn_steps - args.no_l1_epochs)
            )
        else:
            current_alpha = 0
        # compute output
        output_clean = model(image)
        loss = torch.sum((1 - w[index]) * criterion(output_clean, target))
                          
        if with_l1:
            loss += current_alpha * l1_regularization(model)

        optimizer.zero_grad()
        loss.backward()

        optimizer.sign_step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                "Time {3:.2f}".format(
                    epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                )
            )
            start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg


@w_iterative_unlearn
def w_FT(train_loader, model, criterion, optimizer, epoch, args, w, mask=None):
    return w_FT_iter(train_loader, model, criterion, optimizer, epoch, args, w, mask)


@w_iterative_unlearn
def w_FT_l1(train_loader, model, criterion, optimizer, epoch, args, w, mask=None):
    return w_FT_iter(
        train_loader, model, criterion, optimizer, epoch, args, w, mask, with_l1=True
    )

import torch.nn as nn

def reinit_and_freeze_layers(model, k, reinit):
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Get all the layers
    layers = [module for name, module in model.named_modules()]

    # Check if k is valid
    if k > len(layers):
        raise ValueError("k cannot be greater than the number of layers in the model.")

    done = 0
    # Reverse the layers list to start from the last layer
    layers.reverse()
    for layer in layers:
        # Check if we have retrained enough layers
        if done >= k:
            break

        if reinit:
            with torch.no_grad():
                if isinstance(layer, nn.Linear):
                    # Use normal initialization for Linear layers
                    nn.init.normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.Conv2d):
                    # Use xavier initialization for Conv2d layers
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Unfreeze parameters
        for param in layer.parameters():
            param.requires_grad = True

        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            done += 1

@iterative_unlearn
def EU(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    reinit_and_freeze_layers(model, k=1, reinit=True)
    lr = optimizer.param_groups[0]['lr']
    weight_decay = optimizer.param_groups[0]['weight_decay']
    momentum = optimizer.param_groups[0]['momentum']
    
    # Create a new optimizer with the same parameters but only for the parameters that require gradients
    new_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
    return FT_iter(data_loaders, model, criterion, new_optimizer, epoch, args, mask)

@iterative_unlearn
def CF(data_loaders, model, criterion, optimizer, epoch, args, mask=None):

    reinit_and_freeze_layers(model, k=1, reinit=False)
    lr = optimizer.param_groups[0]['lr']
    weight_decay = optimizer.param_groups[0]['weight_decay']
    momentum = optimizer.param_groups[0]['momentum']

    # Create a new optimizer with the same parameters but only for the parameters that require gradients
    new_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
    return FT_iter(data_loaders, model, criterion, new_optimizer, epoch, args, mask)

@w_iterative_unlearn
def w_EU(train_loader, model, criterion, optimizer, epoch, args, w, mask):
    reinit_and_freeze_layers(model, k=1, reinit=True)
    lr = optimizer.param_groups[0]['lr']
    weight_decay = optimizer.param_groups[0]['weight_decay']
    momentum = optimizer.param_groups[0]['momentum']
    
    # Create a new optimizer with the same parameters but only for the parameters that require gradients
    new_optimizer = utils.SignSGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
    return w_FT_iter(train_loader, model, criterion, new_optimizer, epoch, args, w, mask)