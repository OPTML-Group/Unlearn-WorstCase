import os
import arguments.data
import importlib

# HACK: Change isfile


def check_ffcv_available_from_path(path):
    return os.path.isfile(path)

def import_module(name):
    from . import waterbirds
    from . import cifar10
    from . import cifar100
    if name == "waterbirds":
        return waterbirds
    elif name == "cifar10":
        return cifar10
    elif name == "cifar100":
        return cifar100
    else:
        raise Exception(f"{name} dataset not implemented!")

def get_data_module_from_path(path):
    for name in arguments.data.CPU_DATASETS:
        if name in path:
            return import_module(name)
    raise Exception(f"Cannot find cpu data module from path: {path}!")


def get_train_loader_from_path(path, *args, **kwargs):
    module = get_data_module_from_path(path)
    return module.get_train_loader(path, *args, **kwargs)


def get_test_loader_from_path(path, *args, **kwargs):
    module = get_data_module_from_path(path)
    return module.get_test_loader(path, *args, **kwargs)
