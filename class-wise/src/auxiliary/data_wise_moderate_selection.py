import argparse
import os
import sys

import numpy as np
import torch
from fastargs import get_current_config, Param, Section
from fastargs.decorators import param, section

sys.path.append(".")

Section('cfg').params(
    dataset=Param(str, 'imagenet or others?', required=True),
    write_path=Param(str, 'where to save flm class selection file?', required=True),
    interval=Param(float, 'ratio of coreset', default=0.5),
)


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


@section('cfg')
@param('dataset')
@param('write_path')
@param('interval')
def main(dataset, write_path, interval):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    os.makedirs(write_path, exist_ok=True)

    # ========== Data Loader ===========
    if dataset == 'imagenet':
        imagenet_feat_dir = 'data_features/train_imagenet_class_fx'
        imagenet_id_dir = 'data_features/train_imagenet_class_data_id'

        imagenet_feat = torch.load(imagenet_feat_dir)
        imagenet_id = torch.load(imagenet_id_dir)
    else:
        raise ValueError(f"Dataset {dataset} is not supported yet!")

    # get the median feature vector of each class
    print('========== Start Calculating Median Feature Vector ==========')
    class_median_feat = {}
    for key in imagenet_feat.keys():
        class_median_feat[key] = get_median(np.array(imagenet_feat[key]))

    ratio = 0.0
    while ratio <= 0.9999999:
        ratio += interval
        print(f'========== Start Calculating Coreset with ratio {ratio} ==========')
        # get the distance of each feature vector to the median feature vector of its class
        low = 0.5 - ratio / 2
        high = 0.5 + ratio / 2
        all_index = []
        for key in imagenet_feat.keys():
            # print(f'========== Start Calculating distance for class {key} ==========')
            class_distance_score = []
            for i in range(len(imagenet_id[key])):
                class_distance_score.append(calc_distance(imagenet_feat[key][i], class_median_feat[key]))

            # Select the data according to the distance score
            distance_score = np.array(class_distance_score)
            sorted_idx = distance_score.argsort()
            low_idx = round(len(imagenet_id[key]) * low)
            high_idx = round(len(imagenet_id[key]) * high)

            ids = sorted_idx[low_idx:high_idx]

            # print(distance_score)
            for i in range(distance_score.shape[0]):
                if i in ids:
                    all_index.append(imagenet_id[key][i])

            if key % 100 == 0:
                print(key)
                print(len(all_index))

        all_index = torch.as_tensor(np.array(all_index))
        print(all_index.shape)
        file_name = f'{dataset}_Data_wise_Moderate_{ratio:.2f}'

        torch.save(all_index, os.path.join(write_path, file_name))

        print(file_name)


if __name__ == '__main__':
    config = get_current_config()
    parser = argparse.ArgumentParser("FLM class selection")
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate()
    config.summary()
    main()
