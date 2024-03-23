import torch
import utils

def validate(val_loader, model, criterion, args):
    """
    Run evaluation
    """
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()
    try:
        for i, (image, target) in enumerate(val_loader):
            image = image.cuda()
            target = target.cuda()
            # compute output
            with torch.no_grad():
                output = model(image)
                loss = criterion(output, target)
            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                        i, len(val_loader), loss=losses, top1=top1
                    )
                )

        print("valid_accuracy {top1.avg:.3f}".format(top1=top1))


    except:
        for i, (image, target, _) in enumerate(val_loader):
            image = image.cuda()
            target = target.cuda()

            # compute output
            with torch.no_grad():
                output = model(image)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                        i, len(val_loader), loss=losses, top1=top1
                    )
                )

        print("valid_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg
