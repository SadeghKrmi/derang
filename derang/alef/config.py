import json
import logging

import os
from pathlib import Path
from typing import Any


logger = logging.getLogger(__package__)

class Config:
    """Configuration of Model"""
    def __init__(self, config_path: str):
        logger.info(f"loading config: `{config_path}`")
        self.config_path = Path(config_path)
        self.config: dict[str, Any] = slef._load_config()


        self.session_name = self.config["session_name"]
        self.data_dir = Path(self.config["data_directory"])

        self.
