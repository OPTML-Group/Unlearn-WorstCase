import os
import pandas as pd

import sys
sys.path.append("manager")
from core import Experiment
from customized_transforms import pretrain_data_prune_method, _apply_re


def rand_gnep_transfer():
    # for dataset in ['dtd', 'food101', 'oxfordpets', 'stanfordcars', 'sun397', 'ucf101', 'flowers102', 'cifar10', 'cifar100']:
    for dataset in ['cifar10', 'cifar100']:
        for method in ["ff"]:

            if dataset == "cifar100":
                data_path = "../../data/cifar100"
            elif dataset == "cifar10":
                data_path = "../../data/cifar10"
            else:
                data_path = f"../../data/{dataset}/ffcv/train_400_10_90.beton"

            path = "file/experiments/imagenet_transfer_to_downstream"
            
            filt_random = lambda e :(e["config.dataset.train_path"] == data_path) \
                                    and (e["config.network.finetune_method"] == method) \
                                    and (e["config.network.architecture"] == "resnet50") \
                                    and ("random" in pretrain_data_prune_method(e["config.network.pretrained_ckpt"])) \

            filt_grad_norm = lambda e: (e["config.dataset.train_path"] == data_path) \
                                    and (e["config.network.finetune_method"] == method) \
                                    and (e["config.network.architecture"] == "resnet50") \
                                    and ("grad_norm" in pretrain_data_prune_method(e["config.network.pretrained_ckpt"])) \
                                        
            exps = [e for e in map(lambda x: Experiment(os.path.join(path, x)), os.listdir(path))]

            random_exps = [e for e in exps if filt_random(e)]
            random_exps.sort(key=lambda x: int(_apply_re(pretrain_data_prune_method(x["config.network.pretrained_ckpt"]), ["[0-9]+\d*"])[-1]), reverse=True)
            grad_norm_exps = [e for e in exps if filt_grad_norm(e)]
            grad_norm_exps.sort(key=lambda x: int(_apply_re(pretrain_data_prune_method(x["config.network.pretrained_ckpt"]), ["[0-9]+\d*"])[-1]), reverse=True)

            print(len(random_exps), len(grad_norm_exps))

            table_name = [list(map(lambda x: x["name"], random_exps)), list(map(lambda x: x["name"], grad_norm_exps))]
            table_name = pd.DataFrame(table_name, index=["random", "grad_norm"], columns=list(range(50, 1000, 50)))

            table_acc = [list(map(lambda x: 100*x["log.last.best_test_top1"], random_exps)), list(map(lambda x: 100*x["log.last.best_test_top1"], grad_norm_exps))]
            table_acc = pd.DataFrame(table_acc, index=["random", "grad_norm"], columns=list(range(50, 1000, 50)))

            with pd.ExcelWriter(f"file/summaries/{dataset}_rn50.xlsx", mode="a") as writer:
                table_name.to_excel(writer, sheet_name=f"{method}_name")
                table_acc.to_excel(writer, sheet_name=f"{method}_acc")

def resnet50_pretrain():
    path = "file/experiments/imagenet_train_from_scratch"

    exps = [e for e in map(lambda x: Experiment(os.path.join(path, x)), os.listdir(path))]
    filt_resnet50 = lambda e: e["config.network.architecture"] == "resnet50"
    exps = [e for e in exps if filt_resnet50(e)]

    filt_random = lambda e: "random" in e["config.dataset.indices.training"]
    filt_grad_norm = lambda e: "grad_norm" in e["config.dataset.indices.training"]

    random_exps = [e for e in exps if filt_random(e)]
    random_exps.sort(key=lambda x: int(_apply_re(x["config.dataset.indices.training"], ["[0-9]+\d*"])[-1]), reverse=True)
    grad_norm_exps = [e for e in exps if filt_grad_norm(e)]
    grad_norm_exps.sort(key=lambda x: int(_apply_re(x["config.dataset.indices.training"], ["[0-9]+\d*"])[-1]), reverse=True)

    print(len(random_exps), len(grad_norm_exps))

    table_name = [list(map(lambda x: x["name"], random_exps)), list(map(lambda x: x["name"], grad_norm_exps))]
    table_name = pd.DataFrame(table_name, index=["random", "grad_norm"], columns=list(range(50, 1000, 50)))

    table_acc = [list(map(lambda x: x["log.last.best_val_top1"], random_exps)), list(map(lambda x: x["log.last.best_val_top1"], grad_norm_exps))]
    table_acc = pd.DataFrame(table_acc, index=["random", "grad_norm"], columns=list(range(50, 1000, 50)))

    with pd.ExcelWriter(f"file/summaries/random_vs_gnep1_imagenet_rn50.xlsx", mode="w") as writer:
        table_name.to_excel(writer, sheet_name=f"name")
        table_acc.to_excel(writer, sheet_name=f"acc")

if __name__ == "__main__":
    rand_gnep_transfer()