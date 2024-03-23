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
    distributed=Param(BoolAsInt(), 'use distributed training? (0/1)', default=0),
    label_smoothing=Param(float, 'label smoothing parameter', default=0.0),
    scheduler_type=Param(OneOf(['step', 'cyclic', 'cosine', 'cosine_with_warmup']), required=True),
)

Section('train.optimizer').params(
    type=Param(OneOf(['AdamW', 'SGD']), default='SGD'),
    lr=Param(float, required=True),
    weight_decay=Param(float, required=True),
    momentum=Param(float, required=True),
)

Section('train.scheduler.cosine_with_warmup').enable_if(
    lambda cfg: cfg['train.scheduler_type'] == 'cosine_with_warmup'
).params(
    warmup_epoch=Param(int, 'number of warmup epochs', required=True),
    starting_factor=Param(float, 'starting factor', required=True),
)

Section('train.scheduler.step').enable_if(
    lambda cfg: cfg['train.scheduler_type'] == 'step'
).params(
    step_ratio=Param(float, 'learning rate step ratio', required=True),
    step_size=Param(int, 'learning rate step size', required=True),
)

Section('train.scheduler.cyclic').enable_if(
    lambda cfg: cfg['train.scheduler_type'] == 'cyclic'
).params(
    lr_peak_epoch=Param(int, 'epoch at which lr peaks', required=True),
)

Section('train.resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', required=True),
    max_res=Param(int, 'the maximum (starting) resolution', required=True),
    end_ramp=Param(int, 'when to stop interpolating resolution', required=True),
    start_ramp=Param(int, 'when to start interpolating resolution', required=True),
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

Section('distributed').enable_if(
    lambda cfg: cfg['train.distributed']
).params(
    world_size=Param(int, 'number gpus', required=True),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='29500'),
)

class Trainer:

    @param('train.seed')
    @param('train.label_smoothing')
    @param('train.distributed')
    def __init__(self, seed, label_smoothing, distributed, gpu = 0):
        self.gpu_id = gpu
        if distributed:
            self.setup_distributed()
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

    @param('distributed.address')
    @param('distributed.port')
    @param('distributed.world_size')
    def setup_distributed(self, address, port, world_size):
        os.environ['MASTER_ADDR'] = address
        os.environ['MASTER_PORT'] = port

        dist.init_process_group("nccl", rank=self.gpu_id, world_size=world_size)

    @staticmethod
    @section('train.resolution')
    @param('min_res')
    @param('max_res')
    @param('end_ramp')
    @param('start_ramp')
    def get_resolution(epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    @param('network.architecture')
    @param('network.blurpool')
    @param('train.distributed')
    def create_network_and_scaler(self, architecture, blurpool, distributed):
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
        if distributed:
            self.network = torch.nn.parallel.DistributedDataParallel(self.network, device_ids=[self.gpu_id])

    @param('dataset.train_path', 'path')
    @param('dataset.num_workers')
    @param('dataset.in_memory')
    @param('train.batch_size')
    @param('train.distributed')
    @param('dataset.indices.training', 'indices')
    def create_train_loader(self, path, num_workers, in_memory, batch_size, distributed, indices=None):
        if indices is not None:
            indices = torch.load(indices, map_location="cpu")
        res = self.get_resolution(epoch=0)
        self.train_loader, self.decoder = get_train_loader(path, num_workers, in_memory, batch_size, res, self.device,
                                                           indices, distributed = distributed)

    @param('dataset.val_path', 'path')
    @param('dataset.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('train.distributed')
    @param('dataset.indices.testing', 'indices')
    def create_val_loader(self, path, num_workers, batch_size, resolution, distributed, indices=None):
        if indices is not None:
            indices = torch.load(indices, map_location="cpu")
        self.val_loader = get_val_loader(path, num_workers, batch_size, resolution, self.device, indices, 
                                         distributed=distributed)

    @param('train.epoch')
    def get_cosine_scheduler(self, epoch):
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epoch * len(self.train_loader))

    @param('train.epoch')
    @param('train.scheduler.cosine_with_warmup.warmup_epoch')
    @param('train.scheduler.cosine_with_warmup.starting_factor')
    def get_cosine_with_warmup_scheduler(self, epoch, warmup_epoch, starting_factor):
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([
                torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=starting_factor, total_iters=warmup_epoch * len(self.train_loader)),
                torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=(epoch - warmup_epoch) * len(self.train_loader))
            ])
        return scheduler

    @param('train.scheduler.step.step_ratio')
    @param('train.scheduler.step.step_size')
    def get_step_scheduler(self, step_ratio, step_size):
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size * len(self.train_loader),
                                               gamma=step_ratio)

    @param('train.epoch')
    @param('train.scheduler.cyclic.lr_peak_epoch')
    def get_cyclic_scheduler(self, epoch, lr_peak_epoch):
        return torch.optim.lr_scheduler.CyclicLR(
                                                self.optimizer, base_lr=1e-4,
                                                max_lr=self.optimizer.param_groups[0]['lr'],
                                                step_size_up=lr_peak_epoch * len(self.train_loader),
                                                step_size_down=(epoch - lr_peak_epoch) * len(self.train_loader),
                                                cycle_momentum=isinstance(self.optimizer, torch.optim.SGD),
                                            )

    @param('train.optimizer.type')
    @param('train.optimizer.lr')
    @param('train.optimizer.weight_decay')
    @param('train.optimizer.momentum')
    @param('train.scheduler_type')
    def create_optimizer_and_scheduler(self, type, lr, weight_decay, momentum, scheduler_type):
        if type == "AdamW":
            self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.SGD(self.network.parameters(), lr=lr, weight_decay=weight_decay,
                                             momentum=momentum)
        self.scheduler = eval(f'self.get_{scheduler_type}_scheduler()')


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
    @param('logging.save_intermediate_frequency')
    def run(self, epoch, dry_run, path=None, save_intermediate_frequency=None):
        for e in range(self.start_epoch, epoch):
            res = self.get_resolution(e)
            self.decoder.output_size = (res, res)
            train_stats = self.train_loop(e)
            val_stats = self.val_loop()

            if not dry_run and self.gpu_id == 0:
                ckpt = {
                    "state_dicts": {
                        "network": self.network.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                    },
                    "current_epoch": e + 1,
                    "best_acc": self.best_acc,
                    "relative_time": time() - self.start_time,
                }
                if val_stats['top_1'] > self.best_acc:
                    self.best_acc = val_stats['top_1']
                    ckpt['best_acc'] = self.best_acc
                    torch.save(ckpt, os.path.join(path, "checkpoints", "best.ckpt"))
                torch.save(ckpt, os.path.join(path, "checkpoints", "newest.ckpt"))
                if save_intermediate_frequency is not None:
                    if (e + 1) % save_intermediate_frequency == 0:
                        torch.save(ckpt, os.path.join(path, "checkpoints", f"epoch{e}.ckpt"))

                self.log(content={
                    'epoch': e,
                    'train': train_stats,
                    'val': val_stats,
                    'best_val_top1': self.best_acc,
                })

    def train_loop(self, epoch):
        self.network.train()

        iterator = tqdm(self.train_loader, ncols=160) if self.gpu_id == 0 else self.train_loader
        for images, target, _ in iterator:
            ### Training start
            self.optimizer.zero_grad(set_to_none=True)

            with autocast():
                output = self.network(images)
                loss_train = self.loss(output, target)

            self.scaler.scale(loss_train).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            ### Training end

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

@param('train.distributed')
def exec(gpu, distributed):
    Trainer(gpu=gpu)
    if distributed:
        dist.destroy_process_group()

def _exec_wrapper(gpu):
    make_config(gpu!=0)
    exec(gpu)

@param('train.distributed')
@param('distributed.world_size')
def launch(distributed, world_size = None):
    if distributed:
        torch.multiprocessing.spawn(_exec_wrapper, nprocs=world_size, join=True)
    else:
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
