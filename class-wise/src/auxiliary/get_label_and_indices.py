import argparse
import os
import sys

import torch
from fastargs import get_current_config, Param, Section
from fastargs.decorators import param, section
from fastargs.validation import File, Or, Folder
from tqdm import tqdm

sys.path.append(".")

Section('cfg').params(
    data_path=Param(Or(File(), Folder()), 'downstream training data file (.beton) or folder', required=True),
    write_path=Param(str, 'where to save label and indices file?', required=True)
)


@section('cfg')
@param('data_path')
@param('write_path')
def main(data_path, write_path):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    if os.path.isfile(data_path):
        from src.data.ffcv_downstream import get_train_loader
        train_loader, _ = get_train_loader(
            data_path, 12, False, 1024, 224, device,
            decoder_kwargs={
                'scale': (1, 1),
                'ratio': (1, 1),
            },
            flip_probability=0.
        )
    import src.data.utils
    train_loader, _ = src.data.utils.get_train_loader_from_path(data_path, 2, 1024, 224, augments=False)

    pbar = tqdm(train_loader, ncols=120)
    ys = []
    for _, y, i in pbar:
        ys.append(torch.cat([y.unsqueeze(1), i.unsqueeze(1)], axis=1))
    ys = torch.cat(ys).cpu()

    torch.save(ys, write_path)


if __name__ == '__main__':
    config = get_current_config()
    parser = argparse.ArgumentParser("Data Label and Indices Getter")
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate()
    config.summary()
    main()
