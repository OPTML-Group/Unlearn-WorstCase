import argparse
import os
import sys

import matplotlib.pyplot as plt
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
    v=Param(str, 'class-wise selection version', required=True),  # choice =['cmt_median', 'ffmm', 'ffm']
    coreset_ratio=Param(float, 'ratio of coreset', default=0.8),
    reverse=Param(bool, 'reverse the order of class selection', default=False),
)

Section('dataset').params(
    src_train_fx_path=Param(File(), 'imagenet training set feature path?', required=True),
    src_train_id_path=Param(File(), 'imagenet training set data id path?', required=True),
    src_val_id_path=Param(File(), 'imagenet val set feature path?', required=True),
    tgt_train_fx_path=Param(File(), 'downstream training set feature path?', required=True),
)


def count_zeros(x):
    return (x == 0).sum()


def index_min_value(my_list):
    min_value = min(my_list)
    min_index = my_list.index(min_value)
    return min_index


def get_median(features):
    # get the median feature vector of each class
    features = np.array(features)
    class_median = np.median(features, axis=0, keepdims=False)
    return class_median


def get_distance(features):
    class_median = get_median(features)
    distance = np.linalg.norm(features - class_median, axis=1)
    return distance


def calc_distance(x, y):
    x = np.array(x)
    y = np.array(y)
    distance = np.linalg.norm(x - y)
    return distance


def get_coreset_mean_feature(feature, rate, pick_middle=True):
    low = 0.5 - rate / 2
    high = 0.5 + rate / 2

    distance = get_distance(feature)

    sorted_idx = distance.argsort()
    low_idx = round(distance.shape[0] * low)
    high_idx = round(distance.shape[0] * high)

    if pick_middle:
        ids = sorted_idx[low_idx:high_idx]
    else:
        ids = np.concatenate((sorted_idx[:low_idx], sorted_idx[high_idx:]))

    corset_feature = []
    for i in range(distance.shape[0]):
        if i in ids:
            corset_feature.append(feature[i])

    corset_mean_feature = torch.as_tensor(np.array(corset_feature)).mean(dim=0)

    return corset_mean_feature


def get_data_index_from_class(class_id, data_id):
    data_index = []
    for i in range(len(data_id.keys())):
        if i in class_id:
            data_index.extend(data_id[i])
    data_index = torch.tensor(np.array(data_index))
    return data_index


@param('cfg.dataset')
@param('cfg.write_path')
@param('cfg.interval')
@param('cfg.v')
@param('cfg.coreset_ratio')
@param('cfg.reverse')
@param('dataset.src_train_fx_path')
@param('dataset.src_train_id_path')
@param('dataset.src_val_id_path')
@param('dataset.tgt_train_fx_path')
def main(dataset, write_path, interval, v, coreset_ratio,
         src_train_fx_path,
         src_train_id_path,
         src_val_id_path,
         tgt_train_fx_path,
         reverse):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    os.makedirs(write_path, exist_ok=True)
    os.makedirs(os.path.join(write_path, v, dataset), exist_ok=True)

    # ========== Data Loader ===========
    # Load ImageNet Features
    imagenet_feat = torch.load(src_train_fx_path)
    imagenet_id = torch.load(src_train_id_path)
    imagenet_val_id = torch.load(src_val_id_path)

    # ======= Load Downstream Features =======
    support_ds = ['dtd', 'food101', 'oxfordpets', 'stanfordcars', 'sun397', 'ucf101', 'cifar10', 'cifar100', 'waterbirds', 'flowers102']
    if dataset not in support_ds:
        raise ValueError(f"Dataset {dataset} is not supported. Please choose from {support_ds}")
    ds_feat_dir = tgt_train_fx_path
    # ds_id_dir = f'data_features/train_{dataset}_class_data_id'

    ds_feat = torch.load(ds_feat_dir)
    # ds_id = torch.load(ds_id_dir)

    # ========= Class-wise Mean Feature For Source&Target Dataset ======
    mean_imagenet_feat = {}
    for key in imagenet_feat.keys():
        feat = np.array(imagenet_feat[key])
        if v == 'ffmm':
            mean_imagenet_feat[key] = get_coreset_mean_feature(feat, 1 - coreset_ratio)
        elif v == 'cmt' or v == 'ffm' or v == 'cmt_median':
            mean_imagenet_feat[key] = torch.as_tensor(feat).mean(dim=0)
        else:
            raise ValueError(f"Version {v} is not supported. Please choose from ['ffmm', 'ffm', 'cmt']")

    if v == 'ffm' or v == 'ffmm':
        prediction_distribution = torch.zeros(len(mean_imagenet_feat.keys()))

        for ds_key in ds_feat:
            for i in range(len(ds_feat[ds_key])):
                data = ds_feat[ds_key][i]
                distance_score = []
                for key in imagenet_feat.keys():
                    distance_score.append(calc_distance(data, mean_imagenet_feat[key]))
                class_index = index_min_value(distance_score)

                prediction_distribution[class_index] += 1

        prediction_distribution = prediction_distribution.int()
        torch.save(prediction_distribution, os.path.join(write_path, v, dataset, f"{dataset}_ALL_Vote.pth"))

        zero_num = count_zeros(prediction_distribution)
        print('zero_num:', zero_num)

        # Plot the histogram
        # plt.hist(prediction_distribution, bins=torch.max(prediction_distribution).item())
        plt.hist(prediction_distribution, bins=20)
        plt.title(v + " prediction distribution - " + dataset + ' - (zero_num: ' + str(zero_num.item()) + ')')
        plt.xlabel("Mapped Class Frequency")
        plt.ylabel("Class Number")
        plt.show()
        plt.savefig(os.path.join(write_path, v, dataset, f"{dataset}_{v}_result.png"))

    elif v == 'cmt' or v == 'cmt_median':
        # get the median feature vector of each class
        print('========== Start Calculating Class-wise Mean Feature Vector ==========')
        mean_downstream_feat = {}
        for key in ds_feat.keys():
            # mean_downstream_feat[key] = get_median(np.array(ds_feat[key]))
            feat = np.array(ds_feat[key])
            if v == 'cmt':
                mean_downstream_feat[key] = get_coreset_mean_feature(feat, 1 - coreset_ratio)
            elif v == 'cmt_median':
                mean_downstream_feat[key] = get_median(np.array(ds_feat[key]))

        distance_score = {}
        for key in imagenet_feat.keys():
            mean_distance = 0.0
            for key2 in ds_feat.keys():
                mean_distance += calc_distance(mean_imagenet_feat[key], mean_downstream_feat[key2])

            distance_score[key] = mean_distance / len(ds_feat.keys())

        torch.save(distance_score, os.path.join(write_path, v, dataset, f"{dataset}_{v}_ALL_Distance.pth"))

    # ========== Pick Top K Classes ==========
    pick_num = 0
    while pick_num < 1000:

        pick_num += interval

        if v == 'ffm' or v == 'ffmm':
            print(f'Pick Top {pick_num} Classes according to {v}')
            top_classes = torch.topk(prediction_distribution, k=pick_num, largest=not reverse).indices
            selected_data_index = get_data_index_from_class(top_classes, imagenet_id)
            selected_val_data_index = get_data_index_from_class(top_classes, imagenet_val_id)

            if reverse:
                torch.save(top_classes, os.path.join(write_path, v, dataset, f"{dataset}_Last{pick_num}.pth"))
                torch.save(selected_data_index,
                           os.path.join(write_path, v, dataset, f"{dataset}_Last{pick_num}_data_index.pth"))
                torch.save(selected_val_data_index,
                           os.path.join(write_path, v, dataset, f"{dataset}_Last{pick_num}_val_data_index.pth"))
            else:
                torch.save(top_classes, os.path.join(write_path, v, dataset, f"{dataset}_Top{pick_num}.pth"))
                torch.save(selected_data_index,
                           os.path.join(write_path, v, dataset, f"{dataset}_Top{pick_num}_data_index.pth"))
                torch.save(selected_val_data_index,
                           os.path.join(write_path, v, dataset, f"{dataset}_Top{pick_num}_val_data_index.pth"))

        elif v == 'cmt' or v == 'cmt_median':
            sorted_distance_score = sorted(distance_score.items(), key=lambda x: x[1], reverse=reverse)
            selected_class_index = sorted_distance_score[:pick_num]

            selected_class_file = []
            for i in range(len(selected_class_index)):
                selected_class_file.append(selected_class_index[i][0])

            selected_class_file = torch.tensor(np.array(selected_class_file))
            selected_data_index = get_data_index_from_class(selected_class_file, imagenet_id)
            selected_val_data_index = get_data_index_from_class(selected_class_file, imagenet_val_id)

            if reverse:
                file_name = dataset + '_Last' + str(selected_class_file.shape[0])
            else:
                file_name = dataset + '_Top' + str(selected_class_file.shape[0])

            torch.save(selected_class_file, os.path.join(write_path, v, dataset, file_name + '.pth'))
            torch.save(selected_data_index, os.path.join(write_path, v, dataset, file_name + '_data_index.pth'))
            torch.save(selected_val_data_index, os.path.join(write_path, v, dataset, file_name + '_val_data_index.pth'))
            print(file_name)


if __name__ == '__main__':
    config = get_current_config()
    parser = argparse.ArgumentParser("FLM class selection")
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate()
    config.summary()
    main()
