import argparse
from fastargs import get_current_config, Param, Section
from fastargs.decorators import param, section
from fastargs.validation import OneOf, File
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.autograd import grad
from torch.cuda.amp import autocast

import sys
sys.path.append(".")
from src.data.ffcv_downstream import get_train_loader

Section('cfg').params(
    data_path = Param(File(), 'downstream training data file (.beton)', required=True),
    architecture = Param(OneOf(['resnet18', 'resnet50']), required=True),
    pretrained_ckpt = Param(File(), 'pretrained checkpoint path', required=True),
    write_path = Param(str, 'where to save score file?', required=True)
)

@section('cfg')
@param('data_path')
@param('architecture')
@param('pretrained_ckpt')
@param('write_path')
def main(data_path, architecture, pretrained_ckpt, write_path):
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

    state_dict = torch.load(pretrained_ckpt, map_location=device)["state_dicts"]["network"]
    network.load_state_dict(state_dict)
    params_requires_grad = [p for p in network.parameters() if p.requires_grad]

    train_loader, _ = get_train_loader(data_path, 2, False, 4, 160, device, shuffle=False)

    pbar = tqdm(train_loader, desc='Gradient Norm Calculation', ncols=120, total=len(train_loader))
    scores = []
    for x, y, _ in pbar:
        I_N = torch.eye(x.size(0)).to(device)
        with autocast():
            batch_loss = F.cross_entropy(network(x), y, reduction='none')
            jacobian_rows = [torch.cat([g.flatten() for g in grad(batch_loss, params_requires_grad, v, retain_graph=True)]) for v in I_N.unbind()]
            jacobian = torch.stack(jacobian_rows)
        scores.append(jacobian.norm(p=2, dim=1))
    scores = torch.cat(scores).cpu()

    torch.save(scores, write_path)

if __name__ == '__main__':
    config = get_current_config()
    parser = argparse.ArgumentParser("FLM class selection")
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate()
    config.summary()
    main()