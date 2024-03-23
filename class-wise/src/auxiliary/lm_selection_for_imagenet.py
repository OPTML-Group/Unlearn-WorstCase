import argparse
import os
import sys

import torch
from fastargs import get_current_config, Param, Section
from fastargs.decorators import param, section
from fastargs.validation import OneOf, File, ListOfInts, Or, Folder
from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.append(".")

Section('cfg').params(
    data_path=Param(Or(File(), Folder()), 'downstream training data file (.beton) or folder', required=True),
    source_train_label_path=Param(File(), required=True),
    source_val_label_path=Param(File(), required=True),
    architecture=Param(OneOf(['resnet18', 'resnet50']), required=True),
    pretrained_ckpt=Param(File(), 'pretrained checkpoint path', required=True),
    retain_class_nums=Param(ListOfInts(), 'retain classes numbers', required=True),
    reverse=Param(bool, 'reverse frequency ranking?', is_flag=True),
    write_path=Param(str, 'where to save lm class selection file?', required=True),
    return_frequency=Param(bool, 'return frequency before ranking?', is_flag=True),
)


def tensor_a_in_b(a, b):
    a_expanded = a.unsqueeze(1).expand(-1, b.shape[0])
    b_expanded = b.unsqueeze(0).expand(a.shape[0], -1)

    matches = (a_expanded == b_expanded).any(dim=1)
    return matches


@section('cfg')
@param('data_path')
@param('source_train_label_path')
@param('source_val_label_path')
@param('architecture')
@param('pretrained_ckpt')
@param('retain_class_nums')
@param('reverse')
@param('write_path')
@param('return_frequency')
def main(data_path, source_train_label_path, source_val_label_path, architecture, pretrained_ckpt, retain_class_nums,
         reverse, write_path, return_frequency):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    if architecture == "resnet18":
        from torchvision.models import resnet18
        network_init_func = resnet18
    elif architecture == "resnet50":
        from torchvision.models import resnet50
        network_init_func = resnet50
    else:
        raise NotImplementedError(f"{architecture} is not supported")
    network = network_init_func().to(device)
    network.eval()

    state_dict = torch.load(pretrained_ckpt, map_location=device)["state_dicts"]["network"]
    network.load_state_dict(state_dict)
    source_train_labels = torch.load(source_train_label_path, map_location="cpu")
    source_val_labels = torch.load(source_val_label_path, map_location="cpu")

    import src.data.utils
    if src.data.utils.check_ffcv_available_from_path(data_path):
        from src.data.ffcv_downstream import get_train_loader
        train_loader, _ = get_train_loader(
            data_path, 2, 1024, 224, device,
            decoder_kwargs={
                'scale': (1, 1),
                'ratio': (1, 1),
            },
            flip_probability=0.
        )
    else:
        train_loader, _ = src.data.utils.get_train_loader_from_path(data_path, 2, 1024, 224, augments=False)

    pbar = tqdm(train_loader, desc='Inferencing', ncols=120, total=len(train_loader))
    fx = []
    for x, _, _ in pbar:
        x = x.to(device)
        with torch.no_grad(), autocast():
            fx.append(torch.argmax(network(x), dim=-1))
    fx = torch.cat(fx).cpu()

    prediction_distribution = torch.Tensor([(fx == i).sum() for i in range(1000)]).int()

    if return_frequency:
        torch.save(prediction_distribution, f"{write_path}_lm.frequency")
    else:
        for retain_class_num in retain_class_nums:
            top_classes = torch.topk(prediction_distribution, k=retain_class_num, largest=not reverse).indices
            source_train_indices = source_train_labels[tensor_a_in_b(source_train_labels[:, 0], top_classes), 1]
            source_val_indices = source_val_labels[tensor_a_in_b(source_val_labels[:, 0], top_classes), 1]
            torch.save(source_train_indices,
                    f"{write_path}_lm_train_{'bottom' if reverse else 'top'}{retain_class_num}.indices")
            torch.save(source_val_indices,
                    f"{write_path}_lm_val_{'bottom' if reverse else 'top'}{retain_class_num}.indices")


if __name__ == '__main__':
    config = get_current_config()
    parser = argparse.ArgumentParser("LM class selection")
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate()
    config.summary()
    main()
