import copy
import os
import time

import torch
import utils


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def get_optimizer_and_scheduler(model, args):
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )
    return optimizer, scheduler


def train(train_loader, model, criterion, optimizer, epoch, args, mask=None, l1=False):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
   
    for i, (image, target) in enumerate(train_loader):
        if epoch < args.warmup:
            utils.warmup_lr(
                epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
            )

        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)

        loss = criterion(output_clean, target)
        if l1:
            loss = loss + args.alpha * l1_regularization(model)
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


def train_with_rewind(model, optimizer, scheduler, train_loader, criterion, args):
    rewind_state_dict = None
    for epoch in range(args.epochs):
        start_time = time.time()
        train(train_loader, model, criterion, optimizer, epoch, args)

        if (epoch + 1) == args.rewind_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.save_dir, "epoch_{}_rewind_weight.pt".format(epoch + 1)
                ),
            )
            if args.prune_type == "rewind_lt":
                rewind_state_dict = copy.deepcopy(model.state_dict())

        scheduler.step()
        print("one epoch duration:{}".format(time.time() - start_time))

    return rewind_state_dict


def w_train(train_loader, model, criterion, optimizer, epoch, args, w=None, l1=False):
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

        # compute output
        output_clean = model(image)

        loss = torch.sum((1 - w[index]) * criterion(output_clean, target))
        if l1:
            loss = loss + args.alpha * l1_regularization(model)
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