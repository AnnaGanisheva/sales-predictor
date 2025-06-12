from pathlib import Path
from box import ConfigBox
import yaml
from src.utils.logger import logger


def read_yaml(path: Path) -> ConfigBox:
    with open(path, "r") as f:
        config = ConfigBox(yaml.safe_load(f))
        logger.info(f"yaml file: {path} loaded successfully")
    return config
