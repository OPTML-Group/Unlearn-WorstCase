import os
from terminaltables import SingleTable

import sys
sys.path.append("manager")
from core import Experiment

if __name__ == '__main__':

    path = "file/experiments/imagenet_transfer_to_downstream"
    
    def filt(e):
        return e["config.train.epoch"] != e["log.last.epoch"] + 1

    exps = [e for e in map(lambda x: Experiment(os.path.join(path, x)), os.listdir(path))]
    exps = [e for e in exps if filt(e)]

    table = [["names", "path", "epoch"]]
    for exp in exps:
        table.append([
                    exp["name"],
                    exp["config.dataset.train_path"],
                    f"{exp['log.last.epoch']+1}/{exp['config.train.epoch']}",
                    ])
    print(len(table) - 1, "experiments unfinished")
    # table.sort(key=lambda x: int(_apply_re(x[0], ["[0-9]+\d*"])[-1] if x[0] is not None else 1000))
    # table = [["pretrain", "test_acc", "optimizer", "lr"]] + table
    table = SingleTable(table)
    print(table.table)
