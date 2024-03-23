import argparse
import os
import sys

import torch
from fastargs import get_current_config, Param, Section
from fastargs.decorators import param, section
from fastargs.validation import OneOf, Or, Folder
from torch.cuda.amp import autocast
from tqdm import tqdm
import arguments.data

sys.path.append(".")
from fastargs.validation import File

Section('cfg').params(
    data_path=Param(Or(File(), Folder()), 'downstream training data file (.beton) or folder', required=True),
    dataset=Param(OneOf(['imagenet', 'dtd', 'flowers102', 'ucf101', 'food101', 'sun397', 'oxfordpets', 'stanfordcars', 'cifar10', 'cifar100', 'waterbirds']), required=True),
    architecture=Param(OneOf(['resnet18', 'resnet50']), required=True),
    pretrained_ckpt=Param(File(), 'pretrained checkpoint path', required=True),
    write_path=Param(str, 'where to save flm class selection file?', required=True),
    batch=Param(int, 'batch size', default=128),
    class_wise=Param(bool, 'whether to select class-wise', default=False)
)


def tensor_a_in_b(a, b):
    a_expanded = a.unsqueeze(1).expand(-1, b.shape[0])
    b_expanded = b.unsqueeze(0).expand(a.shape[0], -1)

    matches = (a_expanded == b_expanded).any(dim=1)
    return matches


@section('cfg')
@param('dataset')
@param('data_path')
@param('architecture')
@param('pretrained_ckpt')
@param('write_path')
@param('batch')
@param('class_wise')
def main(data_path, dataset, architecture, pretrained_ckpt, write_path, batch, class_wise):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    os.makedirs(write_path.split('/')[0], exist_ok=True)

    # ========== Model Arch ===========
    if architecture == "resnet18":
        from torchvision.models import resnet18
        network_init_func = resnet18
    elif architecture == "resnet50":
        from torchvision.models import resnet50
        network_init_func = resnet50
    else:
        raise NotImplementedError(f"{architecture} is not supported")
    network = network_init_func().to(device)

    # ========== Load Pretrained Model CKPT ============
    state_dict = torch.load(pretrained_ckpt, map_location=device)["state_dicts"]["network"]
    network.load_state_dict(state_dict)

    # ========== Create Hook for feature map of last layer ==========
    features = {}

    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()

        return hook

    feature_check = 'avgpool'
    for n, m in network.named_modules():
        if n == feature_check:
            m.register_forward_hook(get_features('feats'))

    # ========== Data Loader ===========
    if dataset == 'imagenet':
        from src.data.imagenet import get_train_loader_no_preprocess
        train_loader, _ = get_train_loader_no_preprocess(
            data_path, 2, False, batch, 224, device,
            shuffle=False,
            decoder_kwargs={
                'scale': (1, 1),
                'ratio': (1, 1),
            }
        )
    else:
        import src.data.utils
        if src.data.utils.check_ffcv_available_from_path(data_path):
            from src.data.ffcv_downstream import get_train_loader
            train_loader, _ = get_train_loader(
                data_path, 2, False, 1024, 224, device,
                decoder_kwargs={
                    'scale': (1, 1),
                    'ratio': (1, 1),
                },
                flip_probability=0.
            )
        else:
            train_loader, _ = src.data.utils.get_train_loader_from_path(data_path, 2, 1024, 224, augments=False)

    assert dataset in arguments.data.ALL_DATASETS
    class_num = arguments.data.CLASS_NUM[dataset]

    # =========== Start Inferencing ===========
    pbar = tqdm(train_loader, desc=f'Feature Extraction on {dataset}', ncols=120, total=len(train_loader))
    network.eval()

    if class_wise:
        # class-wise file init
        class_fx = {}
        class_data_id = {}

        for i in range(class_num):
            class_fx[i] = []
            class_data_id[i] = []
    else:
        # data-wise file init
        all_fx = []
        all_data_id = []

    for x, y, img_id in pbar:
        x = x.to(device)
        with torch.no_grad(), autocast():
            _ = network(x)

        for j in range(x.size(0)):
            if class_wise:
                # class-wise file
                class_fx[y[j].item()].append(features['feats'][j, :, :, :].flatten().cpu().numpy())
                class_data_id[y[j].item()].append(img_id[j].detach().cpu().item())
            else:
                # Data-wise file
                all_fx.append(features['feats'][j, :, :, :].flatten().cpu().numpy())
                all_data_id.append(img_id[j])

    # ========== Save Files ==========
    if class_wise:
        torch.save(class_fx, f"{write_path}_{dataset}_class_fx")
        torch.save(class_data_id, f"{write_path}_{dataset}_class_data_id")
        # print(class_data_id)
    else:
        torch.save(all_fx, f"{write_path}_{dataset}_all_fx")
        torch.save(all_data_id, f"{write_path}_{dataset}_all_data_id")


if __name__ == '__main__':
    config = get_current_config()
    parser = argparse.ArgumentParser("FLM class selection")
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate()
    config.summary()
    main()
