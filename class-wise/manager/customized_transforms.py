import re
import os
from typing import List

import sys
sys.path.append("manager")
from core import Experiment

def _apply_re(value, expr: List[str]):
    if isinstance(expr, str):
        expr = [expr]
    for e in expr:
        pattern = re.compile(e)
        value = pattern.findall(value)
    return value

def _find_exp(x):
    while True:
        try:
            pretrain_exp = Experiment(x)
            break
        except (NotADirectoryError, FileNotFoundError):
            x = os.path.split(x)[0]
    return pretrain_exp


def data_prune_ratio_indomain(x):
    return int(_apply_re(x, ["\_[0-9]+\d*\.", "[0-9]+\d*"]))

def data_prune_ratio_flm(x):
    return int(_apply_re(x, ["top[0-9]+\d*\.", "[0-9]+\d*"]))

def pretrain_data_prune_ratio_flm(x):
    """
    This function finds number in the form of xxx_650_xxx, where the number has '_' before and after it.
    Note if the name in the file is 650, this will return 350 as there is a semantic transfer from the
    'retain ratio' to 'pruning ratio'
    """
    return 1000 - int(_apply_re(x, ["\_[0-9]+\d*\_", "[0-9]+\d*"]))


def pretrain_data_prune_ratio(x):
    e = _find_exp(x)
    return 1000 - int(_apply_re(e["config.dataset.indices.training"], ["\_[0-9]+\d*\.", "[0-9]+\d*"]))


def pretrain_data_prune_method(x):
    e = _find_exp(x)
    if e["config.dataset.indices.training"] is not None:
        return str(e["config.dataset.indices.training"])
    else:
        return str(1000)

def convert_to_percent(x):
    return 100*x