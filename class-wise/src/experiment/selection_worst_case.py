import argparse
import json
import os
import sys
from datetime import datetime
from time import time

import numpy as np
import torch
import torchmetrics
from fastargs import get_current_config, Param, Section
from fastargs.decorators import param, section
from fastargs.validation import OneOf, File, Folder, BoolAsInt
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.distributed as dist
from tqdm import tqdm

sys.path.append(".")
from src.tools.training import apply_blurpool, MeanScalarMetric
from src.tools.misc import set_seed
from src.data.imagenet import get_train_loader, get_val_loader

from src.algorithm.sign_sgd import SignSGD
from src.algorithm.bi_section import bisection

Section('exp').params(
    identifier=Param(str, 'experiment identifier', default=None)
)

Section('network').params(
    architecture=Param(OneOf(['resnet18', 'resnet50', 'resnet101', 'resnet152', 'vit_b_16']), required=True),
    blurpool=Param(BoolAsInt(), 'use blurpool? (0/1)', default=0),
)

Section('dataset').params(
    train_path=Param(File(), required=True),
    val_path=Param(File(), required=True),
    num_workers=Param(int, 'the number of workers', default=12),
    in_memory=Param(BoolAsInt(), 'does the dataset fit in memory? (0/1)', default=0),
    prune=Param(BoolAsInt(), 'is the dataset pruned? (0/1)', default=0),
)

Section('dataset.indices').enable_if(
    lambda cfg: cfg['dataset.prune']
).params(
    training=Param(File(), required=True),
    testing=Param(File(), required=False),
)

Section('train').params(
    seed=Param(int, required=True),
    epoch=Param(int, required=True),
    batch_size=Param(int, required=True),
    label_smoothing=Param(float, 'label smoothing parameter', default=0.0)
)

Section('train.optimizer').params(
    type=Param(OneOf(['SignSGD', 'AdamW', 'SGD']), default='SignSGD'),
    lr=Param(float, required=True),
    weight_decay=Param(float, required=True),
    momentum=Param(float, required=True),
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'the batch size for validation', default=512),
    resolution=Param(int, 'final resized validation image size', default=224),
    lr_tta=Param(BoolAsInt(), 'should do lr flipping/avging at test time? (0/1)', default=0),
)

Section('logging', 'how to log stuff').params(
    dry_run=Param(bool, 'use log or not', is_flag=True),
    path=Param(Folder(), 'resume path, if new experiment leave blank', default=None),
    save_intermediate_frequency=Param(int, 'save extra checkpoints for every t epoch'),
)

Section('blo', 'bi-level optimization for data selection').params(
    pretrained_ckpt=Param(File(), 'pretrained checkpoint path', default="~/DP4TL/file/experiments/imagenet_train_from_scratch/origin/checkpoints/best.ckpt"),
    w_lr=Param(float, 'learning rate of upper_level', default=1e3),
    gamma=Param(float, 'the ratio of l2 regularation term', default=1e-4),
)

class Trainer:

    @param('train.seed')
    @param('train.label_smoothing')
    def __init__(self, seed, label_smoothing, gpu = 0):
        self.gpu_id = gpu
        self.device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
        torch.cuda.set_device(self.device)
        set_seed(seed)
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.create_train_loader()
        self.create_val_loader()
        self.create_network_and_scaler()
        self.create_optimizer_and_scheduler()
        self.initialize_metrics()
        self.resume()
        self.run()

    @param('network.architecture')
    @param('network.blurpool')
    def create_network_and_scaler(self, architecture, blurpool):
        self.scaler = GradScaler()
        if architecture == "resnet18":
            from torchvision.models import resnet18
            network = resnet18(weights=None)
        elif architecture == "resnet50":
            from torchvision.models import resnet50
            network = resnet50(weights=None)
        elif architecture == "resnet101":
            from torchvision.models import resnet101
            network = resnet101(weights=None)
        elif architecture == "resnet152":
            from torchvision.models import resnet152
            network = resnet152(weights=None)
        elif architecture == "vit_b_16":
            from torchvision.models import vit_b_16
            network = vit_b_16(weights=None)
        else:
            raise NotImplementedError(f"{architecture} is not supported")

        if blurpool:
            apply_blurpool(network)
        
        self.network = network.to(device=self.device)

    @param('dataset.train_path', 'path')
    @param('dataset.num_workers')
    @param('dataset.in_memory')
    @param('train.batch_size')

    @param('dataset.indices.training', 'indices')
    def create_train_loader(self, path, num_workers, in_memory, batch_size, indices=None):
        if indices is not None:
            indices = torch.load(indices, map_location="cpu")
        res = 224
        self.train_loader, self.decoder = get_train_loader(path, num_workers, in_memory, batch_size, res, self.device,
                                                           indices)

    @param('dataset.val_path', 'path')
    @param('dataset.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('dataset.indices.testing', 'indices')
    def create_val_loader(self, path, num_workers, batch_size, resolution, indices=None):
        if indices is not None:
            indices = torch.load(indices, map_location="cpu")
        self.val_loader = get_val_loader(path, num_workers, batch_size, resolution, self.device, indices)

    @param('train.optimizer.type')
    @param('train.optimizer.lr')
    @param('train.optimizer.weight_decay')
    @param('train.optimizer.momentum')
    def create_optimizer_and_scheduler(self, type, lr, weight_decay, momentum):
        if type == "SignSGD":
            self.optimizer = SignSGD(self.network.parameters(), lr=lr)
        elif type == "AdamW":
            self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.SGD(self.network.parameters(), lr=lr, weight_decay=weight_decay,
                                             momentum=momentum)

    def initialize_metrics(self):
        self.train_meters = {
            'top_1': torchmetrics.Accuracy(task='multiclass', num_classes=1000).to(self.device),
            'loss': MeanScalarMetric().to(self.device)
        }
        self.val_meters = {
            'top_1': torchmetrics.Accuracy(task='multiclass', num_classes=1000).to(self.device),
            'top_5': torchmetrics.Accuracy(task='multiclass', num_classes=1000, top_k=5).to(self.device),
            'loss': MeanScalarMetric().to(self.device)
        }
        self.start_time = time()
        self.best_acc = 0.
        self.start_epoch = 0

    @param('logging.path')
    def resume(self, path=None):
        try:
            ckpt = torch.load(os.path.join(path, "checkpoints", "newest.ckpt"), map_location=self.device)
            for key, val in ckpt["state_dicts"].items():
                eval(f"self.{key}.load_state_dict(val)")
            self.best_acc = ckpt["best_acc"]
            self.start_epoch = ckpt["current_epoch"]
            self.start_time -= ckpt["relative_time"]
        except FileNotFoundError:
            os.makedirs(os.path.join(path, "checkpoints"), exist_ok=False)
        except TypeError:
            pass

    @param('logging.path')
    def log(self, content, path):
        print(f'=> Log: {content}')
        cur_time = time()
        path = os.path.join(path, 'log.json')
        stats = {
            'timestamp': cur_time,
            'relative_time': cur_time - self.start_time,
            **content
        }
        if os.path.isfile(path):
            with open(path, 'r') as fd:
                old_data = json.load(fd)
            with open(path, 'w') as fd:
                fd.write(json.dumps(old_data + [stats]))
                fd.flush()
        else:
            with open(path, 'w') as fd:
                fd.write(json.dumps([stats]))
                fd.flush()

    @param('train.epoch')
    @param('logging.dry_run')
    @param('logging.path')
    @param('blo.model_path')
    @param('blo.w_lr')
    @param('blo.gamma')
    def run(self, epoch, dry_run, w_lr, gamma, model_path, path=None):

        # Initialize the data selection weight tensor
        w = torch.zeros(1000).to(self.device).requires_grad_(False)  # Assuming there are 1000 classes
        _, indices = torch.topk(torch.rand(1000), int(0.1 * 1000))  # Randomly select 10% of the classes
        w[indices] = 1  # Set the selected classes to 1

        for u in range(20):
            state_dict = torch.load(model_path, map_location=self.device)["state_dicts"]["network"]
            self.network.load_state_dict(state_dict)

            for e in range(self.start_epoch, epoch):
                self.decoder.output_size = (224, 224)

                # Lower level optimization
                train_stats = self.train_loop(e, w)
                val_stats = self.val_loop()

                if not dry_run and self.gpu_id == 0:
                    ckpt = {
                        "state_dicts": {
                            "network": self.network.state_dict(),
                            "optimizer": self.optimizer.state_dict()
                        },
                        "current_epoch": e + 1,
                        "best_acc": self.best_acc,
                        "relative_time": time() - self.start_time,
                    }
                    torch.save(ckpt, os.path.join(path, "checkpoints", "newest.ckpt"))


                self.log(content={
                    'epoch': e,
                    'train': train_stats,
                    'val': val_stats,
                    'best_val_top1': self.best_acc,
                })

            # Upper level optimization
            _, w_grad_tensor = self.upper_loop()
            w -= w_lr * (torch.tensor(w_grad_tensor, dtype=torch.float64).cuda() + gamma * 2 * w)
            w = bisection(w, int(0.1 * 1000 * 0.9))

            upper_loss = torch.sum(w * w_grad_tensor)

            # Log the upper_loss and w for the current epoch
            self.log(content={
                'epoch': u,
                'upper_loss': upper_loss.item(),
                'w': w.detach().cpu().numpy().tolist(),
            })

            # Save the data selection weights
            torch.save(w, os.path.join(path, f"select_weight_{u}.pth.tar"))

    def upper_loop(self):
        self.network.eval()

        iterator = tqdm(self.train_loader, ncols=120) if self.gpu_id == 0 else self.train_loader

        # Initialize a tensor to store the sum of loss for each class
        class_loss_sum = torch.zeros(1000).to(self.device)  # Assuming there are 1000 classes

        for images, target, _ in iterator:
            with torch.no_grad(), autocast():
                output = self.network(images)

                for k in ['top_1', 'top_5']:
                    self.val_meters[k](output, target)

                # Compute the loss for each class separately
                for i in range(1000):  # Assuming there are 1000 classes
                    class_output = output[target == i]
                    class_target = target[target == i]

                    if class_output.nelement() > 0:  # Check if there are samples of this class in the batch
                        class_loss = self.loss(class_output, class_target) / len(class_output)
                        class_loss_sum[i] += class_loss.item()

                # Compute the total loss for the batch
                total_loss_val = self.loss(output, target)
                self.val_meters['loss'](total_loss_val)
                stats = {k: m.compute().item() for k, m in self.val_meters.items()}

            if self.gpu_id == 0:
                names = ['acc', 'loss']
                values = [f"{stats['top_1']:.3f}", f"{stats['loss']:.3f}"]

                msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                iterator.set_description(msg)

        if self.gpu_id == 0:
            [meter.reset() for meter in self.val_meters.values()]

        # Return the stats and the sum of loss for each class
        return stats, class_loss_sum

    def train_loop(self, epoch, w):
        self.network.train()
        loss_function = torch.nn.CrossEntropyLoss(reduction='none')

        iterator = tqdm(self.train_loader, ncols=160) if self.gpu_id == 0 else self.train_loader
        for images, target, _ in iterator:
            ### Training start
            self.optimizer.zero_grad(set_to_none=True)

            with autocast():
                output = self.network(images)
                # Compute the loss for each sample without reduction
                loss_per_sample = loss_function(output, target)  # shape: (N,)
                # Get the weights for each sample based on their class
                weights = 1 - w[target]  # shape: (N,)
                # Compute the weighted sum of the losses
                loss_train = (weights * loss_per_sample).sum()

            self.scaler.scale(loss_train).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.train_meters['top_1'](output, target)
            self.train_meters['loss'](loss_train)
            stats = {k: m.compute().item() for k, m in self.train_meters.items()}

            group_lrs = []
            for _, group in enumerate(self.optimizer.param_groups):
                group_lrs.append(f'{group["lr"]:.3e}')

            if self.gpu_id == 0:
                names = ['ep', 'lrs', 'acc', 'loss']
                values = [epoch, group_lrs, f"{stats['top_1']:.3f}", f"{stats['loss']:.3f}"]

                msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                iterator.set_description(msg)

        if self.gpu_id == 0:
            [meter.reset() for meter in self.train_meters.values()]
        return stats
    

    @param('validation.lr_tta')
    def val_loop(self, lr_tta):
        self.network.eval()

        iterator = tqdm(self.val_loader, ncols=120) if self.gpu_id == 0 else self.val_loader

        for images, target, _ in iterator:
            with torch.no_grad(), autocast():
                output = self.network(images)
                if lr_tta:
                    output += self.network(torch.flip(images, dims=[3]))
            
            for k in ['top_1', 'top_5']:
                self.val_meters[k](output, target)

            loss_val = self.loss(output, target)
            self.val_meters['loss'](loss_val)
            stats = {k: m.compute().item() for k, m in self.val_meters.items()}

            if self.gpu_id == 0:
                names = ['acc', 'loss']
                values = [f"{stats['top_1']:.3f}", f"{stats['loss']:.3f}"]

                msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                iterator.set_description(msg)

        if self.gpu_id == 0:
            [meter.reset() for meter in self.val_meters.values()]
        return stats

def exec(gpu):
    Trainer(gpu=gpu)

def _exec_wrapper(gpu):
    make_config(gpu!=0)
    exec(gpu)
    
def launch():
    _exec_wrapper(0)

def make_config(quiet=False):
    config = get_current_config()
    parser = argparse.ArgumentParser("Imagenet Train from scratch")
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)

    if config['logging.path'] is not None:
        assert not config['logging.dry_run'], "dry run can not accept resume path!"
        config.collect_config_file(os.path.join(config['logging.path'], 'config.json'))
        config.validate()
    else:
        config.validate()
        if config['exp.identifier'] is not None:
            file_name = config['exp.identifier']
        else:
            file_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

        path = os.path.join("file", "experiments", os.path.basename(__file__.split('.')[0]), file_name)

        for attr in dir(config):
            if not attr.startswith('__'):
                print(f'{attr}: {getattr(config, attr)}')

        if not config['logging.dry_run'] and not quiet:
            os.makedirs(path, exist_ok=False)
            config.collect({'logging': {'path': path}})

        if not quiet:
            config.summary()

if __name__ == "__main__":
    make_config(True)
    launch()