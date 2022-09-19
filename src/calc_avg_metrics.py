from collections import defaultdict
from os import PathLike
from sys import argv
from numpy import mean
from pathlib import Path
import json
from itertools import product

def calc_avg(model_or_base_dir: PathLike, split: str, metrics: tuple) -> str:
    agg_metrics = defaultdict(list)

    for single_f1_res_dir in model_or_base_dir.iterdir():
        if f"_{split}_" in single_f1_res_dir.name:
            with open(single_f1_res_dir / f"{split}_metrics.json") as f:
                metrics_json = json.load(f)
            for metric in metrics:
                agg_metrics[metric].append(metrics_json[f'{split}_{metric}'])
    return [mean(agg_metrics[m]) for m in metrics]

def main(base_dir: str) -> None:
    root = Path(base_dir).parent # root is "num_ex=XX"
    metrics = 'precision', 'recall', 'f1'
    dirs = [d.name for d in root.iterdir()]
    
    for split in 'dev', 'test':
        if any(split in d.name for d in (root / dirs[0]).iterdir()):
            with open(root.parent / f'avg_{split}_metrics.txt', 'a') as f:
                # Headers
                if f.tell() < 1:
                    f.write('\t'.join(["#ex"] + \
                        [f"{m[0].upper()} ({d[0]})" for d, m in product(dirs, metrics)]) + '\n')

                values = [str(round(v, 4)) for model_or_base in dirs for v in \
                    calc_avg(root / model_or_base, split, metrics)]
                num_train = str(root).split('_')[-1].split('=')[1]
                f.write('\t'.join([num_train] + values) + '\n')


if __name__ == "__main__":
    main(*argv[1:])
