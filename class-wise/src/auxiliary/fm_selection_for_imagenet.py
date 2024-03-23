import argparse
import sys

import numpy as np
import torch
from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from fastargs.validation import File

sys.path.append(".")

Section('cfg').params(
    dataset=Param(str, 'imagenet or others?', required=True),
    write_path=Param(str, 'where to save flm class selection file?', required=True),
    interval=Param(int, 'interval number', default=50),
)

Section('dataset').params(
    src_train_fx_path=Param(File(), 'imagenet training set feature path?', required=True),
    src_train_id_path=Param(File(), 'imagenet training set data id path?', required=True),
    src_val_id_path=Param(File(), 'imagenet val set feature path?', required=True),
    tgt_train_fx_path=Param(File(), 'downstream training set feature path?', required=True),
)


def count_zeros(x):
    return (x == 0).sum()


def calc_distance(x, y):
    distance = torch.norm(x - y, p=2)
    return distance


def get_data_index_from_class(class_id, data_id):
    data_index = []
    for i in range(len(data_id.keys())):
        if i in class_id:
            data_index.extend(data_id[i])
    data_index = torch.tensor(data_index)
    return data_index


@param('cfg.dataset')
@param('cfg.write_path')
@param('cfg.interval')
@param('dataset.src_train_fx_path')
@param('dataset.src_train_id_path')
@param('dataset.src_val_id_path')
@param('dataset.tgt_train_fx_path')
def main(dataset, write_path, interval,
         src_train_fx_path,
         src_train_id_path,
         src_val_id_path,
         tgt_train_fx_path):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # ======= Load Downstream Features =======
    support_ds = ['dtd', 'food101', 'oxfordpets', 'stanfordcars', 'sun397', 'ucf101', 'cifar10', 'cifar100', 'waterbirds',
                  'flowers102']
    if dataset not in support_ds:
        raise ValueError(f"Dataset {dataset} is not supported. Please choose from {support_ds}")

    # ========== Data Loader ===========
    # Load ImageNet Features
    imagenet_feat = torch.load(src_train_fx_path, map_location=device)
    imagenet_id = torch.load(src_train_id_path, map_location=device)
    imagenet_val_id = torch.load(src_val_id_path, map_location=device)
    ds_feat = torch.load(tgt_train_fx_path, map_location=device)

    # ========= Class-wise Mean Feature For Source & Target Dataset ======
    mean_imagenet_feat = {}
    for key in imagenet_feat.keys():
        feat = torch.tensor(np.array(imagenet_feat[key])).to(device)
        mean_imagenet_feat[key] = feat.mean(dim=0)

    imagenet_class_num = len(mean_imagenet_feat.keys())
    prediction_distribution = torch.zeros(imagenet_class_num, device=device)

    for ds_key in ds_feat:
        data_num_per_class = len(ds_feat[ds_key])
        for i in range(data_num_per_class):
            data = torch.tensor(np.array(ds_feat[ds_key][i]), device=device)
            distance_score = torch.zeros(imagenet_class_num, device=device).requires_grad_(False)
            for key in imagenet_feat.keys():
                distance_score[i] = calc_distance(data, mean_imagenet_feat[key])
            class_index = distance_score.argmin()
            prediction_distribution[class_index] += 1

    prediction_distribution = prediction_distribution.int()
    zero_num = count_zeros(prediction_distribution)
    print('zero_num:', zero_num)

    # ========== Pick Top K Classes ==========
    pick_num = 0
    while pick_num < 1000:
        pick_num += interval

        print(f'Pick Top {pick_num} Classes according to fm')
        top_classes = torch.topk(prediction_distribution, k=pick_num, largest=True).indices
        selected_data_index = get_data_index_from_class(top_classes, imagenet_id)
        selected_val_data_index = get_data_index_from_class(top_classes, imagenet_val_id)
        torch.save(top_classes, write_path + f"{dataset}_fm_Top{pick_num}_class_index.pth")
        torch.save(selected_data_index, write_path + f"{dataset}_fm_Top{pick_num}_train_data_index.pth")
        torch.save(selected_val_data_index, write_path + f"{dataset}_fm_Top{pick_num}_val_data_index.pth")


if __name__ == '__main__':
    config = get_current_config()
    parser = argparse.ArgumentParser("FM class selection")
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate()
    config.summary()
    main()
