import numpy as np

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

from .const import IMAGENET_MEAN, IMAGENET_STD

def get_train_loader(path, num_workers, in_memory, batch_size, res, device, indices=None, shuffle=True, distributed=False):
    decoder = RandomResizedCropRGBImageDecoder((res, res))

    image_pipeline = [
        decoder,
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]
    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(device, non_blocking=True)
    ]
    identifier_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(device, non_blocking=True)
    ]

    if not distributed:
        order = OrderOption.QUASI_RANDOM if shuffle else OrderOption.SEQUENTIAL
    else:
        order = OrderOption.RANDOM if shuffle else OrderOption.SEQUENTIAL
    train_loader = Loader(path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=order,
                    indices=indices,
                    os_cache=in_memory,
                    drop_last=False,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline,
                        'identifier': identifier_pipeline
                    },
                    distributed=distributed
                )
    return train_loader, decoder

def get_train_loader_no_preprocess(path, num_workers, in_memory, batch_size, res, device, indices=None, shuffle=True, decoder_kwargs={}):
    decoder = RandomResizedCropRGBImageDecoder((res, res), **decoder_kwargs)

    image_pipeline = [
        decoder,
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]
    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(device, non_blocking=True)
    ]
    identifier_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(device, non_blocking=True)
    ]

    order = OrderOption.QUASI_RANDOM if shuffle else OrderOption.SEQUENTIAL
    train_loader = Loader(path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=order,
                    indices=indices,
                    os_cache=in_memory,
                    drop_last=False,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline,
                        'identifier': identifier_pipeline
                    },
                )
    return train_loader, decoder


def get_val_loader(path, num_workers, batch_size, res, device, indices=None, distributed=False):

    cropper = CenterCropRGBImageDecoder((res, res), ratio=224/256)
    image_pipeline = [
        cropper,
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]
    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(device, non_blocking=True)
    ]
    identifier_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(device, non_blocking=True)
    ]

    val_loader = Loader(path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=OrderOption.SEQUENTIAL,
                    drop_last=False,
                    indices=indices,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline,
                        'identifier': identifier_pipeline
                    },
                    distributed=distributed
                )
    return val_loader

# from ffcv.fields.base import Field
# class RandomLabel(Field):
#     def __init__(self, num_classes):
#         self.num_classes = num_classes

#     def __call__(self, label):
#         return np.random.randint(0, self.num_classes)

    # def accept_field(self, field):
    #     return self(field)

import torch as ch
from dataclasses import replace
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
# class RandomLabel(Operation):
#     def __init__(self, num_classes):
#         self.num_classes = num_classes

#     def __call__(self, label):
#         return np.random.randint(0, self.num_classes)

#     def accept_field(self, field):
#         return self(field)
    
class RandomLabel(Operation):
    """Convert from Numpy array to PyTorch Tensor."""
    def __init__(self):
        super().__init__()

    def generate_code(self) -> Callable:
        def to_tensor(inp, dst):
            return ch.randint(0, 1000, inp.shape)
        return to_tensor

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        new_dtype = ch.from_numpy(np.empty((), dtype=previous_state.dtype)).dtype
        return replace(previous_state, jit_mode=False, dtype=new_dtype), None

def get_relabel_loader(path, num_workers, in_memory, batch_size, res, device, indices=None, shuffle=True, distributed=False):
    decoder = RandomResizedCropRGBImageDecoder((res, res))

    image_pipeline = [
        decoder,
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]
    label_pipeline = [
        IntDecoder(),
        RandomLabel(),
        # ToTensor(),
        Squeeze(),
        ToDevice(device, non_blocking=True)
    ]
    identifier_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(device, non_blocking=True)
    ]

    if not distributed:
        order = OrderOption.QUASI_RANDOM if shuffle else OrderOption.SEQUENTIAL
    else:
        order = OrderOption.RANDOM if shuffle else OrderOption.SEQUENTIAL
    train_loader = Loader(path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=order,
                    indices=indices,
                    os_cache=in_memory,
                    drop_last=False,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline,
                        'identifier': identifier_pipeline
                    },
                    distributed=distributed
                )
    return train_loader, decoder