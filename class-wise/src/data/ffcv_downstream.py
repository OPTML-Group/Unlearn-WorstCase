import numpy as np
from ffcv.fields.basics import IntDecoder
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage

from .const import IMAGENET_MEAN, IMAGENET_STD


def get_train_loader(path, num_workers, batch_size, res, device, indices=None, shuffle=True, decoder_kwargs={},
                     flip_probability=0.5, in_memory=False):
    decoder = RandomResizedCropRGBImageDecoder((res, res), **decoder_kwargs)

    image_pipeline = [
        decoder,
        RandomHorizontalFlip(flip_probability),
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


def get_train_loader_no_preprocess(path, num_workers, batch_size, res, device, indices=None, shuffle=True,
                                   decoder_kwargs={}, in_memory=False):
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


def get_test_loader(path, num_workers, batch_size, res, device, indices=None):
    cropper = CenterCropRGBImageDecoder((res, res), ratio=224 / 256)
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

    test_loader = Loader(path,
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
                         )
    return test_loader
