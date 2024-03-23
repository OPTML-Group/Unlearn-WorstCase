import argparse
import json
import os

import pandas as pd


def format_print(float_list):
    formatted_list = ["{:.2f}".format(num) for num in float_list]
    print(" ".join(formatted_list))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--path', type=str, default="file/experiments/imagenet_transfer_to_downstream/")
    p.add_argument('--output', type=str, required=True)
    p.add_argument('--pattern', type=str, required=True, help="dataset_num_ff")
    p.add_argument('--dataset', nargs="+", type=str, default=[])
    p.add_argument('--epoch-num', type=int, default=49)
    p.add_argument('--range', type=str, default="50:951:50")
    args = p.parse_args()

    assert os.path.isdir(args.path), 'exp_path should be path to a folder that contains all the experiments!'
    assert "dataset" in args.pattern and "num" in args.pattern

    dataset_list = ['dtd', 'food101', 'oxfordpets', 'stanfordcars', 'sun397', 'ucf101', 'cifar10', 'cifar100', 'waterbirds',
                    'imagenet', 'flowers102']
    for dataset in args.dataset:
        assert dataset in dataset_list
    range_setting = [int(num) for num in args.range.split(":")]
    assert (len(range_setting) == 3)
    print(range_setting)
    num_list = range(range_setting[0], range_setting[1], range_setting[2])
    print(num_list)

    test_result_key = 'best_test_top1'
    failure_flag = False
    total_data = []
    for dataset in args.dataset:
        result = []
        for num in num_list:
            folder_name = args.pattern.replace("dataset", dataset).replace("num", str(1000 - num))
            result_path = os.path.join(args.path, folder_name, 'log.json')
            f = open(result_path, 'r')
            data = json.load(f)
            current_epoch = data[-1]["epoch"]
            if current_epoch < args.epoch_num:
                failure_flag = True
                print(f"Failure: for folder {folder_name}, there is only {current_epoch} epoch, please resume!")
            result.append(data[-1][test_result_key] * 100)
        format_print(result)
        total_data.append(result)

    if not failure_flag:
        with pd.ExcelWriter(args.output, mode='w') as writer:
            total_data = pd.DataFrame(total_data, index=args.dataset, columns=num_list)
            total_data.to_excel(writer)
