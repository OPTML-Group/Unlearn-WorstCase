import os
import pandas as pd

import sys
sys.path.append("manager")
from core import Experiment
from customized_transforms import pretrain_data_prune_method, _apply_re


if __name__ == '__main__':
    # dataset = "cifar10"
    # method = "ff"
    for dataset in ["cifar10", "cifar100"]:
        for method in ["ff", "lp"]:

            if dataset == "cifar100":
                data_path = "../../data/cifar100"
            elif dataset == "cifar10":
                data_path = "../../data/cifar10"
            else:
                data_path = f"../../data/{dataset}/ffcv/train_400_10_90.beton"

            path = "file/experiments/imagenet_transfer_to_downstream"
            
            filt_random = lambda e :(e["config.dataset.train_path"] == data_path) \
                                    and (e["config.network.finetune_method"] == method) \
                                    and ("random" in pretrain_data_prune_method(e["config.network.pretrained_ckpt"])) \
                                    # and (e["config.train.optimizer.type"] == "SGD") \

            filt_grad_norm = lambda e: (e["config.dataset.train_path"] == data_path) \
                                    and (e["config.network.finetune_method"] == method) \
                                    and ("grad_norm" in pretrain_data_prune_method(e["config.network.pretrained_ckpt"])) \
                                    # and (e["config.train.optimizer.type"] == "SGD") \
                                        
            filt_flm = lambda e: (e["config.dataset.train_path"] == data_path) \
                                    and (e["config.network.finetune_method"] == method) \
                                    and (f"{dataset}_flm" in pretrain_data_prune_method(e["config.network.pretrained_ckpt"])) \

            exps = [e for e in map(lambda x: Experiment(os.path.join(path, x)), os.listdir(path))]

            random_exps = [e for e in exps if filt_random(e)]
            random_exps.sort(key=lambda x: int(_apply_re(pretrain_data_prune_method(x["config.network.pretrained_ckpt"]), ["[0-9]+\d*"])[-1]), reverse=True)
            grad_norm_exps = [e for e in exps if filt_grad_norm(e)]
            grad_norm_exps.sort(key=lambda x: int(_apply_re(pretrain_data_prune_method(x["config.network.pretrained_ckpt"]), ["[0-9]+\d*"])[-1]), reverse=True)
            flm_exps = [e for e in exps if filt_flm(e)]
            flm_exps.sort(key=lambda x: int(_apply_re(pretrain_data_prune_method(x["config.network.pretrained_ckpt"]), ["[0-9]+\d*"])[-1]), reverse=True)

            print(len(random_exps), len(grad_norm_exps), len(flm_exps))

            def get_lp_acc(exps):
                return [e["log.last.best_test_top1"]*100 for e in exps]

            table = [get_lp_acc(random_exps), get_lp_acc(grad_norm_exps), get_lp_acc(flm_exps)]
            table = pd.DataFrame(table, index=["random", "grad_norm", "flm"], columns=list(range(50, 1000, 50))[::-1])
            try:
                with pd.ExcelWriter(f"file/summaries/{dataset}.xlsx", mode="a") as writer:
                    table.to_excel(writer, sheet_name=method)
            except FileNotFoundError:
                with pd.ExcelWriter(f"file/summaries/{dataset}.xlsx", mode="w") as writer:
                    table.to_excel(writer, sheet_name=method)
