import os
import glob
import pandas as pd
from utils import arguments
from logger import logger
import torch
from config import Config
from torch.utils.data import Dataset, DataLoader

def transformer(text):
    ## Write custom transformer to apply on text
    return text


class PersianKasrehDataset(Dataset):
    def __init__(self, config, datapath, transform=None):
        self.datapath = datapath
        self.transform = transform
        self.text_encoder = config.text_encoder
        
        self.data = []

        if os.path.exists(datapath):
            with open(datapath, 'r', encoding='utf-8') as file:
                for line in file:
                    self.data.append(line.strip())
        else:
            raise ValueError(f'file {datapath} does not exist, Are you sure you run preprocessor.py?')
        logger.info(f'The number of text lines for {datapath} set is {self.__len__()}')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = self.data[index]
        if self.transform:
            text = self.transform(text)
        
        sentence_vector, boundary_vect, diacritics_vector = self.text_encoder.extractor(text)
        return {
            'sentence': torch.tensor(sentence_vector, dtype=torch.long),
            'boundary': torch.tensor(boundary_vect, dtype=torch.long),
            'diacritics': torch.tensor(diacritics_vector, dtype=torch.long),
        }


# Collate function to pad sequences, PyTorch expects each batch of tensors to have the same sequence length
def collate_fn(batch):
    sentences = [item['sentence'] for item in batch]
    boundaries = [item['boundary'] for item in batch]
    diacritics = [item['diacritics'] for item in batch]

    # Pad sequences
    sentences_padded = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True, padding_value=0)
    boundaries_padded = torch.nn.utils.rnn.pad_sequence(boundaries, batch_first=True, padding_value=0)
    diacritics_padded = torch.nn.utils.rnn.pad_sequence(diacritics, batch_first=True, padding_value=0)
    
    return {
        'sentence': sentences_padded,
        'boundary': boundaries_padded,
        'diacritics': diacritics_padded
    }


def load_training_data(config):
    datapath = os.path.join(config.config['directory'], 'corpus', 'train.txt')
    dataset = PersianKasrehDataset(config, datapath, transform=transformer)
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=collate_fn, shuffle=True)
    return train_loader

def load_validation_data(config):
    datapath = os.path.join(config.config['directory'], 'corpus', 'validation.txt')
    dataset = PersianKasrehDataset(config, datapath, transform=transformer)
    validation_loader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=collate_fn, shuffle=True)
    return validation_loader

def load_test_data(config):
    datapath = os.path.join(config.config['directory'], 'corpus', 'test.txt')
    dataset = PersianKasrehDataset(config, datapath, transform=transformer)
    test_loader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=collate_fn, shuffle=True)
    return test_loader



if __name__ == '__main__':
    args = arguments().parse_args()
    config_path = args.config
    config = Config(config_path)

    # load the 1st element of the 1st batch for demonstration purpose
    train_loader = load_training_data(config)
    batch = next(iter(train_loader))

    # Access specific elements in the batch
    sentence = batch['sentence']
    boundary = batch['boundary']
    diacritics = batch['diacritics']
    
    # To fetch a single item (first item) from each tensor in the batch
    single_sentence = sentence[0]
    single_boundary = boundary[0]
    single_diacritics = diacritics[0]

    print("Single sentence:", single_sentence)
    print("Single boundary:", single_boundary)
    print("Single diacritics:", single_diacritics)