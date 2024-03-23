import argparse
import os
import pandas as pd

import sys
sys.path.append("manager")
from core import Experiment, parse_path_transform_condition, parse_path_transform
import customized_transforms

def get_transform(key):
    return (lambda x: x) if key == 'same' else customized_transforms.__dict__[key]

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ep', type=str, help='experiment folder')
    p.add_argument('--cons', nargs='+', type=str, help='conditions for all target experiments to satisfy', default=[])
    p.add_argument('--bg', type=str, help='big grouping path@transform:conditions')
    p.add_argument('--sg', type=str, help='small grouping path@transform:conditions')    
    p.add_argument('--hor', type=str, help='horizontal axis path@transform')
    p.add_argument('--ver', type=str, help='vertical axis path@transform')
    p.add_argument('--wp', type=str, help='where to write the xlsx file')
    args = p.parse_args()

    assert os.path.isdir(args.ep), 'ep should be path to a folder that contains all the experiments!'

    exps = [e for e in map(lambda x: Experiment(os.path.join(args.ep, x)), os.listdir(args.ep))]
    for con in args.cons:
        p, t, c = parse_path_transform_condition(con)
        t = get_transform(t)
        exps = [e for e in exps if e.filt(p, c[0], t)]

    if args.bg is not None:
        bgp, bgt, bgcs = parse_path_transform_condition(args.bg)
        bgt = get_transform(bgt)
        grouped_exps = [[e for e in exps if e.filt(bgp, bgc, bgt)] for bgc in bgcs]
    else:
        grouped_exps = [exps]
        bgcs = ["all"]

    if args.sg is not None:
        sgp, sgt, sgcs = parse_path_transform_condition(args.sg)
        sgt = get_transform(sgt)
        grouped_exps = list(map(lambda el: [[e for e in el if e.filt(sgp, sgc, sgt)] for sgc in sgcs], grouped_exps))
    else:
        grouped_exps = list(map(lambda el: [el], grouped_exps))
        sgcs = ["all"]

    hor_p, hor_t = parse_path_transform(args.hor)
    hor_t = get_transform(hor_t)
    get_hor = lambda e: e.get(hor_p, hor_t)

    hs = []
    for bel in grouped_exps:
        for sel in bel:
            sel.sort(key=get_hor)
            hs.append(list(map(get_hor, sel)))
    for i in range(len(hs)-1): assert hs[i] == hs[i+1], f'horizontal value must be the same across groups, got {hs[i]} and {hs[i+1]}'

    ver_p, ver_t = parse_path_transform(args.ver)
    ver_t = get_transform(ver_t)
    with pd.ExcelWriter(args.wp, mode='w') as writer:
        for bel, bgc in zip(grouped_exps, bgcs):
            table = [list(map(lambda e: e.get(ver_p, ver_t), el)) for el in bel] 
            table = pd.DataFrame(table, index=sgcs, columns=hs[0])
            table.to_excel(writer, bgc)
            