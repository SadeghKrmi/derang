import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from derang.logger import _LOGGER
from derang.utils import extract_haraqat

LOAD_NUM_WORKERS = 0


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


class DiacritizationDataset(Dataset):
    """
    The Diacritization dataset loader, the assumption in __getitem__ is that, data is preprocessed.
    """
    
    def __init__(self, config, list_ids, data):
        self.list_ids = list_ids
        self.data = data
        self.text_encoder = config.text_encoder
        self.config = config
        
    def __len__(self):
        return len(self.list_ids)
    
    
    def __getitem__(self, index):
        id = self.list_ids[index]
        if self.config.config["is_data_preprocessed"]:
            data = self.data.iloc[id]
            inputs = torch.Tensor(self.text_encoder.input_to_sequence(data[1]))
            targets = torch.Tensor(self.text_encoder.target_to_sequence(data[2].split(self.config["diacritics_separator"])))
            return inputs, targets, data[0]
        
        data = self.data[id]
        data = self.text_encoder.clean(data)
        
        text, inputs, diacritics = extract_haraqat(data)
        inputs = torch.Tensor(self.text_encoder.input_to_sequence("".join(inputs)))
        diacritics = torch.Tensor(self.text_encoder.target_to_sequence(diacritics))

        return inputs, diacritics, text
    
def collate_fn(data):
    """
    padding the input and output sequences
    """
    
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths
    
    data.sort(key=lambda x: len(x[0]), reverse=True)
    
    src_seqs, trg_seqs, original = zip(*data)
    
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)
    
    batch = {
        "original": original,
        "src": src_seqs,
        "target": trg_seqs,
        "lengths": torch.LongTensor(src_lengths)
    }
    
    return batch


def load_training_data(config, **loader_parameters):
    if config.config["is_data_preprocessed"]:
        path = os.path.join(config.data_dir, "train.csv")
        train_data = pd.read_csv(path, encoding="utf-8", sep=config.config["data_separator"], header=None)
        training_set = DiacritizationDataset(config, train_data.index, train_data)
    else:
        path = os.path.join(config.data_dir, "train.txt")
        with open(path, encoding="utf8") as file:
            train_data = file.readlines()
            train_data = [text for text in train_data if len(text) <= config.config["max_len"]]
        training_set = DiacritizationDataset(config, [idx for idx in range(len(train_data))], train_data)
    
    loader_parameters.setdefault("batch_size", config.config["batch_size"])
    loader_parameters.setdefault("shuffle", True)
    loader_parameters.setdefault("num_workers", LOAD_NUM_WORKERS)
    training_iterator = DataLoader(training_set, collate_fn=collate_fn, **loader_parameters)
    
    _LOGGER.info(f"Length of training iterator = {len(training_iterator)}")
    
    return training_iterator


def load_test_data(config, **loader_parameters):
    if config.config["is_data_preprocessed"]:
        path = os.path.join(config.data_dir, "test.csv")
        test_data = pd.read_csv(path, encoding="utf-8", sep=config.config["data_separator"], header=None)
        test_dataset = DiacritizationDataset(config, test_data.index, test_data)
    else:
        test_file_name = "test.txt"
        path = os.path.join(config.data_dir, test_file_name)
        with open(path, encoding="utf8") as file:
            test_data = file.readlines()
        test_data = [text for text in test_data if len(text) <= config.config["max_len"]]
        test_dataset = DiacritizationDataset(config, [idx for idx in range(len(test_data))], test_data)
        
    loader_parameters.setdefault("batch_size", config.config["batch_size"])
    loader_parameters.setdefault("num_workers", LOAD_NUM_WORKERS)
    test_iterator = DataLoader(test_dataset, collate_fn=collate_fn, **loader_parameters)
    
    _LOGGER.info(f"Length of test iterator = {len(test_iterator)}")
    
    return test_iterator



def load_validation_data(config, **loader_parameters):
    if config.config["is_data_preprocessed"]:
        path = os.path.join(config.data_dir, "val.csv")
        valid_data = pd.read_csv(path, encoding="utf-8", sep=config.config["data_separator"], header=None)
        valid_dataset = DiacritizationDataset(config, valid_data.index, valid_data)
    else:
        path = os.path.join(config.data_dir, "val.txt")
        with open(path, encoding="utf8") as file:
            valid_data = file.readlines()
        valid_data = [text for text in valid_data if len(text) <= config.config["max_len"]]
        valid_dataset = DiacritizationDataset(config, [idx for idx in range(len(valid_data))], valid_data)
        
    loader_parameters.setdefault("batch_size", config.config["batch_size"])
    loader_parameters.setdefault("num_workers", LOAD_NUM_WORKERS)
    valid_iterator = DataLoader(valid_dataset, collate_fn=collate_fn, **loader_parameters)
    
    _LOGGER.info(f"Length of valididation iterator = {len(valid_iterator)}")
    
    return valid_iterator

