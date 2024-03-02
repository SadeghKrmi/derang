import json
from logger import _LOGGER
from pathlib import Path
from typing import Any

class Config:
    """Derang Model Configuration Parser"""
    def __init__(self, config_path: str):
        _LOGGER.info(f"Reading the config file from {config_path}...")
        self.config_path = Path(config_path)
        self.config: dict[str, Any] = self._load_config()
        _LOGGER.info(f"Parsing the config into json \n{self.config}")
        self.session_name = self.config["session_name"]
        self.data_dir = self.config["data_directory"]
        
        
    def _load_config(self):
        with open(self.config_path, "rb") as jconfig:
            return json.load(jconfig)
    
    def __getitem(self, key):
        return self.config[key]
    
    def __contains(self, key):
        return key in self.config
    
    def get(self, key, default=None):
        return self.config.get(key, default)
    
    