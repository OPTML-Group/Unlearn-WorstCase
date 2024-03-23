"""
    setup model and datasets
"""


import copy
import os
import random

# from advertorch.utils import NormalizeByChannelMeanStd
import shutil
import sys
import time

import numpy as np
import torch
from dataset import *
from dataset import TinyImageNet
from models import *
from torchvision import transforms

import torch.optim as optim
from torch.autograd import grad
from torch.autograd.functional import jacobian

import matplotlib.pyplot as plt

__all__ = [
    "setup_model_dataset",
    "AverageMeter",
    "warmup_lr",
    "save_checkpoint",
    "setup_seed",
    "accuracy",
]

def warmup_lr(epoch, step, optimizer, one_epoch_step, args):
    overall_steps = args.warmup * one_epoch_step
    current_steps = epoch * one_epoch_step + step

    lr = args.lr * current_steps / overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p["lr"] = lr


def save_checkpoint(
    state, is_SA_best, save_path, pruning, filename="checkpoint.pth.tar"
):
    filepath = os.path.join(save_path, str(pruning) + filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(
            filepath, os.path.join(save_path, str(pruning) + "model_SA_best.pth.tar")
        )


def load_checkpoint(device, save_path, pruning, filename="checkpoint.pth.tar"):
    filepath = os.path.join(save_path, str(pruning) + filename)
    if os.path.exists(filepath):
        print("Load checkpoint from:{}".format(filepath))
        return torch.load(filepath, device)
    print("Checkpoint not found! path:{}".format(filepath))
    return None


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dataset_convert_to_train(dataset):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset.transform = train_transform
    dataset.train = False


def dataset_convert_to_test(dataset, args=None):
    if args.dataset == "TinyImagenet":
        test_transform = transforms.Compose([])
    elif args.dataset == "trans_cifar10":
        test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]
        )
    elif args.dataset == "celeba":
        test_transform = transforms.Compose(
            # [
            #     transforms.Resize(256),
            #     transforms.CenterCrop(224),
            #     transforms.ToTensor(),
            # ]
            [
                transforms.CenterCrop((178, 178)),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]  
        )
    else:
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset.transform = test_transform
    dataset.train = False


def setup_model_dataset(args):
    if args.dataset == "cifar10":
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )
        train_set, valid_set, test_set = cifar10_datasets(data_dir=args.data)
        print("dataset length: ", len(train_set), len(valid_set), len(test_set))    


    elif args.dataset == "cifar100":
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]
        )
        train_set, valid_set, test_set = cifar100_datasets(data_dir=args.data)

    elif args.dataset == "celeba":
        classes = 2
        # https://github.com/ssagawa/overparam_spur_corr/blob/09df90db3bae9a9686e509152c20a88e22670bba/data/celebA_dataset.py#L94
        normalization = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        train_set, valid_set, test_set = celeba_datasets(data_dir=args.data)

    elif args.dataset == "TinyImagenet":
        classes = 200
        normalization = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        train_set, valid_set, test_set = TinyImageNet(args).datasets(
            seed=args.seed
        )

    if args.imagenet_arch or args.dataset == 'celeba':
        model = model_dict[args.arch](num_classes=classes, imagenet=True)
    elif args.arch == 'swin_t':
        model = swin_t(window_size=4,num_classes=10,downscaling_factors=(2,2,2,1))
    else:
        model = model_dict[args.arch](num_classes=classes)

    model.normalize = normalization
    print(model)
    return model, train_set, valid_set, test_set


def setup_model_indexdataset(args):
    if args.dataset == "cifar10":
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )
        train_set, valid_set, test_set = cifar10_index_datasets(data_dir=args.data)
        print("dataset length: ", len(train_set), len(valid_set), len(test_set))    


    elif args.dataset == "cifar100":
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]
        )
        train_set, valid_set, test_set = cifar100_index_datasets(data_dir=args.data)


    elif args.dataset == "TinyImagenet":
        classes = 200
        normalization = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        train_set, valid_set, test_set = TinyImageNet(args).index_datasets(
            seed=args.seed
        )

    elif args.dataset == 'celeba':
        classes = 2
        normalization = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_set, valid_set, test_set = celeba_index_datasets(data_dir=args.data)

    if args.imagenet_arch or args.dataset == 'celeba':
        model = model_dict[args.arch](num_classes=classes, imagenet=True)
    elif args.arch == 'swin_t':
        model = swin_t(window_size=4,num_classes=10,downscaling_factors=(2,2,2,1))
    else:
        model = model_dict[args.arch](num_classes=classes)

    model.normalize = normalization
    print(model)
    return model, train_set, valid_set, test_set


def setup_seed(seed):
    # print("setup random seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return self.normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return "mean={}, std={}".format(self.mean, self.std)

    def normalize_fn(self, tensor, mean, std):
        """Differentiable version of torchvision.functional.normalize"""
        # here we assume the color channel is in at dim=1
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(mean).div(std)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def update_w(
    origin_train_set, 
    w=None, 
    class_to_replace: int = None, 
    num_indexes_to_replace: int = None,
    seed: int = 1,
    batch_size:int = 128,
    shuffle: bool = False,
    only_mark: bool = True,
    args=None
):
    def replace_loader_dataset(
        dataset, batch_size=batch_size, seed=seed, shuffle=True
    ):
        setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )
    
    train_full_loader = replace_loader_dataset(
        origin_train_set, batch_size=batch_size, seed=seed, shuffle=shuffle
    )

    train_set = copy.deepcopy(origin_train_set)
    if w is None:
        if class_to_replace is not None:
            indexes = replace_class(
                train_set,
                class_to_replace,
                num_indexes_to_replace=num_indexes_to_replace,
                seed=seed - 1,
                only_mark=only_mark,
            )

            # binary
            w = torch.zeros(len(train_set))
            w[indexes] = 1

            # uniform
            # w = torch.ones(len(train_set)) / len(train_set)

    else:
        indexes = torch.where(w == 1)[0].tolist()
        replace_indexes(train_set, indexes, seed, only_mark)

    if args.dataset == "cifar10" or args.dataset == "cifar100" or args.dataset == "trans_cifar10":
        forget_dataset = copy.deepcopy(train_set)
        marked = forget_dataset.targets < 0
        forget_dataset.data = forget_dataset.data[marked]
        forget_dataset.targets = -forget_dataset.targets[marked] - 1
        forget_loader = replace_loader_dataset(
            forget_dataset, batch_size=batch_size, seed=seed, shuffle=shuffle
        )

        retain_dataset = copy.deepcopy(train_set)
        marked = retain_dataset.targets >= 0
        retain_dataset.data = retain_dataset.data[marked]
        retain_dataset.targets = retain_dataset.targets[marked]
        retain_loader = replace_loader_dataset(
            retain_dataset, batch_size=batch_size, seed=seed, shuffle=shuffle
        )

    elif args.dataset == "TinyImagenet":
        forget_dataset = copy.deepcopy(train_set)
        marked = forget_dataset.targets < 0
        forget_dataset.imgs = forget_dataset.imgs[marked]
        forget_dataset.targets = -forget_dataset.targets[marked] - 1
        forget_loader = replace_loader_dataset(
            forget_dataset, batch_size=batch_size, seed=seed, shuffle=shuffle
        )

        retain_dataset = copy.deepcopy(train_set)
        marked = retain_dataset.targets >= 0
        retain_dataset.imgs = retain_dataset.imgs[marked]
        retain_dataset.targets = retain_dataset.targets[marked]
        retain_loader = replace_loader_dataset(
            retain_dataset, batch_size=batch_size, seed=seed, shuffle=shuffle
        )

    elif args.dataset == "celeba":
        forget_dataset = copy.deepcopy(train_set)
        marked = forget_dataset.labels < 0
        forget_dataset.filenames = np.array(forget_dataset.filenames)[marked].tolist()

        forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(
            forget_dataset, batch_size=batch_size, seed=seed, shuffle=shuffle
        )

        retain_dataset = copy.deepcopy(train_set)
        marked = retain_dataset.labels >= 0
        retain_dataset.filenames = np.array(retain_dataset.filenames)[marked].tolist()
        retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(
            retain_dataset, batch_size=batch_size, seed=seed, shuffle=shuffle
        )
    return train_full_loader, forget_loader, retain_loader, w,


class SignSGD(optim.SGD):
    def __init__(self, params, lr, momentum, weight_decay):
        super().__init__(params, lr, momentum, weight_decay)

    def sign_step(self):
        """Performs a single optimization step using the sign of gradients."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.sign()
                p.data.add_(d_p, alpha=-group['lr'])


def weights_init(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)


# torch
def bisection(a, eps, xi=1e-5, ub=1, max_iter=1e2):
    mu_l = torch.min(a - 1)
    mu_u = torch.max(a)
    iter_count = 0
    mu_a = (mu_u + mu_l) / 2  

    while torch.abs(mu_u - mu_l) > xi:
        # print(torch.abs(mu_u - mu_l))
        mu_a = (mu_u + mu_l) / 2
        gu = torch.sum(torch.clamp(a - mu_a, 0, ub)) - eps
        gu_l = torch.sum(torch.clamp(a - mu_l, 0, ub)) - eps

        if gu == 0 or iter_count >= max_iter:
            break
        if torch.sign(gu) == torch.sign(gu_l):
            mu_l = mu_a
        else:
            mu_u = mu_a

        iter_count += 1

    upper_S_update = torch.clamp(a - mu_a, 0, ub)
    return upper_S_update


def compute_gradients(images, targets, model, optimizer, criterion, sign):
    gradients = []
    for image, target in zip(images, targets):
        optimizer.zero_grad()

        output = model(image.unsqueeze(0))  # Unsqueeze to add a batch dimension
        loss = criterion(output, target.unsqueeze(0))  # Unsqueeze to add a batch dimension

        grad_loss = grad(loss, model.parameters())
        gradients.append([g.detach().cpu().numpy() for g in grad_loss])
        if sign:
            optimizer.sign_step()
        else:
            optimizer.step()

    return gradients