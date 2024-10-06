from trainer import train, w_train

from .impl import iterative_unlearn, w_iterative_unlearn


@iterative_unlearn
def retrain(data_loaders, model, criterion, optimizer, epoch, args, mask):
    retain_loader = data_loaders["retain"]
    return train(retain_loader, model, criterion, optimizer, epoch, args, mask)

@w_iterative_unlearn
def w_retrain(train_loader, model, criterion, optimizer, epoch, args, w, mask):
    return w_train(train_loader, model, criterion, optimizer, epoch, args, w, mask)