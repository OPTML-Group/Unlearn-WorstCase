import numpy as np
from typing import List, Union

def parse_path_transform(expr: str):
    path_and_transform = expr.split('@', 2)
    if len(path_and_transform) == 1:
        path_and_transform.append('same')
    elif len(path_and_transform) != 2:
        raise ValueError(f'invalid path and transform {path_and_transform}')
    return path_and_transform

def parse_path_transform_condition(expr: str):
    path_and_transform, conditions = expr.split(':', 2)
    path, transform = parse_path_transform(path_and_transform)
    conditions = conditions.split(',')
    return path, transform, conditions
