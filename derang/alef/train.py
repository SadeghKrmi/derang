import torch
import random
import argparse
import numpy as np
from pathlib import Path
from derang.logger import _LOGGER
from derang.config import Config


def train_arg_parser():
    _LOGGER.info("Running the argument parser in train module...")
    parser = argparse.ArgumentParser(
        description = "argument parser for training module."
    )
    parser.add_argument("--config", type=str, required=True, help="Model configiguration path")
    choices = ["gpu", "cpu"]
    parser.add_argument("--accelerator", type=str, default=choices[0], choices=choices, help="set the accelerator to gpu or cpu.")
    parser.add_argument("--devices", type=int, default=1, help="number of available devices.")
    parser.add_argument("--seed", type=str, default=1234, help="seed number, set to a value for the purpose of reproducibiliy.")
    parser.add_argument("--continue-from", type=str, help="checkpoint to continue the taining.")
    return parser
    
def main():
    args = train_arg_parser().parse_args()
    config = Config(args.config)
    
    # set the seed for random functions
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logs_root_directory = Path(config.config["logs_root_directory"])
    logs_root_directory.mkdir(parents=True, exist_ok=True)
    _LOGGER.info(f"logs are being stored in {logs_root_directory} directory.")
    
    # @todo: implement derang/alef/model.py first then proceed here.

    return


if __name__ == "__main__":
    main()