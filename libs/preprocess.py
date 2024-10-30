import os
import glob
import random
import shutil
from utils import arguments
from config import Config
from pathlib import Path
from logger import logger
from typing import Any, Optional
from language import PERSIAN_CHARS


def take_out_samples(lines, n) -> (list, list): # type: ignore
    sample = random.sample(list(lines), n)
    return (list(set(lines).difference(sample)), list(sample))


if __name__ == '__main__':
    args = arguments().parse_args()
    config_path = args.config
    config = Config(config_path)

    directory = config.__get__('directory')
    logger.info(f'preprocessing the dataset corpus text under {directory}')

    outpath = os.path.join(directory, 'corpus')
    if os.path.exists(outpath):
        shutil.rmtree(outpath)

    os.mkdir(outpath)

    max_workers = max(4, os.cpu_count() * 2)
    files = glob.glob(f'{directory}/*.csv')

    lines = []
    for fl in files:
        with open(fl, 'r', encoding='utf-8') as file:
            for line in file:
                lines.append(line.strip())

    unique_chars_list_from_datasets = set()
    for line in lines:
        for char in line:
            unique_chars_list_from_datasets.add(char)

    difference = unique_chars_list_from_datasets - PERSIAN_CHARS
    if difference:
        logger.error(f'the following characters include in the dataset which are not defined in langauge.py')
        for line in lines:
            diffs = {element for element in difference if element in line}
            if diffs:
                print(diffs, '------>', line)


    random.shuffle(lines)
    n_lines = len(lines)

    n_test = round(n_lines * 0.01)
    lines, test_lines = take_out_samples(lines, n_test)

    n_val = round(n_lines * 0.02)
    lines, val_lines = take_out_samples(lines, n_val)

    # Update the size of train set after creation of val and test sets
    n_lines = len(lines)
    
    logger.info(f'writing training set, test set and validation set into {outpath}/')
    logger.info(f'training set contains {n_lines}, test set {n_test} and validation set {n_val} sentences')
    Path(os.path.join(outpath, 'train.txt')).write_text("\n".join(lines), encoding="utf-8", newline="\n")
    Path(os.path.join(outpath, 'test.txt')).write_text("\n".join(test_lines), encoding="utf-8", newline="\n")
    Path(os.path.join(outpath, 'validation.txt')).write_text("\n".join(val_lines), encoding="utf-8", newline="\n")
