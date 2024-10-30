import os
import json
from logger import logger
from encoder import TextEncoder

class Config:
    def __init__(self, config_path: str):
        logger.info(f"Reading the config file from {config_path} directory")
        self.config_path = os.path.abspath(config_path)
        self.config = self.__load_config__()
        self.text_encoder = TextEncoder()

    def __load_config__(self):
        with open(self.config_path, 'rb') as jconfig:
            return json.load(jconfig)
        
    def __getitem__(self, key):
        return self.config[key]
    
    def __get__(self, key, default=None):
        return self.config.get(key, default)
    
    def __contains__(self, key):
        return key in self.config