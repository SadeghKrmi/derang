import json
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from derang.logger import _LOGGER
from derang.config import Config
from derang.alef.model import AlefModel
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.plugins.precision import MixedPrecisionPlugin
from derang.utils import find_last_checkpoint
from derang.dataset import load_test_data, load_training_data, load_validation_data

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
    parser.add_argument("--test", action="store_true", help="Run the test after training")
    parser.add_argument("--debug", action="store_true", help="Use fast dev mode of ligntning")
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
    
    model = AlefModel(config)
    checkpoint_save_callbacks = []
    if config.config["model_save_steps"]:
        checkpoint_save_callbacks.append(ModelCheckpoint(every_n_train_steps = config.config["model_save_steps"]))
    if config.config["model_save_epochs"]:
        checkpoint_save_callbacks.append(ModelCheckpoint(every_n_epochs = config.config["model_save_steps"]))
        
    wer_early_stop_callback = EarlyStopping(monitor="WER", min_delta=0.00, patience=15, mode="min", strict=True)
    
    plugins = []
    if config.config["use_mixed_precision"]:
        _LOGGER.info("Configuting automatic mixed precision")
        mp = (MixedPrecisionPlugin("16-mixed", device="cuda", scaler=torch.cuda.amp.GradScaler()) if args.accelerator == "gpu" else MixedPrecisionPlugin("bf16-mixed", device="cpu"))
        plugins = [mp,]
        
    
    trainer = Trainer(
        accelerator=args.accelerator, 
        devices=args.devices, 
        check_val_every_n_epoch=config.config["evaluate_epochs"],
        callbacks=[wer_early_stop_callback, *checkpoint_save_callbacks],
        plugins=plugins,
        max_epochs = config.config["max_epochs"],
        enable_progress_bar = True,
        enable_model_summary = True,
        fast_dev_run = args.debug,
        log_every_n_steps = 10,
        default_root_dir = logs_root_directory
    )
    
    if args.continue_from:
        if args.continue_from == "last":
            checkpoint_filename, epoch, step = find_last_checkpoint(config.config["logs_root_directory"])
            args.continue_from = checkpoint_filename
            _LOGGER.info(f"Automatically using checkpoint last checkpoint from: epoch={epoch} - step={step}")
        
        _LOGGER.info(f"Continueing training from checkpoint: {args.continue_from}")
        trainer.ckpt_path = args.continue_from
        
    if args.test:
        _LOGGER.info("Testing is starting ...")
        test_loader = load_test_data(config)
        trainer.test(model, test_loader)
    else:
        train_loader, val_loader = load_training_data(config), load_validation_data(config)
        inference_config_path = logs_root_directory.joinpath("inference-config.json")
        
        with open(inference_config_path, "w", encoding="utf-8", newline="\n") as file:
            json.dump(config.get_inference_config(), file, ensure_ascii=False, indent=2)
        _LOGGER.info(f"Writing inference config to file: `{inference_config_path}`")
        _LOGGER.info("Training loop starting...")
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()