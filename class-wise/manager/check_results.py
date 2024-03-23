import os
from terminaltables import SingleTable

import sys
sys.path.append("manager")
from core import Experiment
from customized_transforms import pretrain_data_prune_method, _apply_re


if __name__ == '__main__':

    path = "file/experiments/imagenet_transfer_to_downstream"
    for dataset in ['dtd', 'food101', 'oxfordpets', 'stanfordcars', 'sun397', 'ucf101', 'flowers102',]: #'cifar10', 'cifar100']:
        for method in ["ff"]:
            def filt(e):
                return (e["config.dataset.train_path"] == f"../../data/{dataset}/ffcv/train_400_10_90.beton") \
                                            and (e["config.network.finetune_method"] == method) \
                                            and (e["config.network.architecture"] == "resnet50") \
                                            and ("random" in pretrain_data_prune_method(e["config.network.pretrained_ckpt"])) \
                                            
            exps = [e for e in map(lambda x: Experiment(os.path.join(path, x)), os.listdir(path))]
            exps = [e for e in exps if filt(e)]

            table = []
            for exp in exps:
                table.append([
                            pretrain_data_prune_method(exp["config.network.pretrained_ckpt"]),
                            exp["config.network.finetune_method"],
                            exp["log.last.best_test_top1"],
                            exp["name"],
                            ])
            table.sort(key=lambda x: int(_apply_re(x[0], ["[0-9]+\d*"])[-1] if x[0] is not None else 1000))
            table = [["pretrain", "finetune_method", "test_acc", "name"]] + table
            table = SingleTable(table, f"{dataset}_{method}")
            if len(exps) != 19:
                print(table.table)
