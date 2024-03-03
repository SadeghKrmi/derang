import os
import random
from derang.logger import _LOGGER

from lightning.pytorch import LightningModule

class AlefModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        
        # @todo: first need to introduce config.len_target_symbols etc.
        hyperparams = {
            
        }