import logging
import pathlib

import colorlog
import yaml


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        logger.handlers = []

    handler = logging.StreamHandler()
    log_colors = {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "magenta",
    }
    if logger_name == "__main__":
        string_format = "%(log_color)s%(levelname)s%(reset)s: %(asctime)s -:::- %(message)s"
        formatter = colorlog.ColoredFormatter(string_format, datefmt="%Y-%m-%d %H:%M:%S", log_colors=log_colors)
    else:
        string_format = "%(log_color)s%(levelname)s%(reset)s: %(asctime)s -:::- %(name)s  - %(message)s"
        formatter = colorlog.ColoredFormatter(string_format, datefmt="%Y-%m-%d %H:%M:%S", log_colors=log_colors)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def get_project_root():
    """Find the project root directory based on the module's location.

    Returns:
        str: The absolute path to the project root directory.
    """
    current_path = pathlib.Path(__file__).resolve()
    potential_root = current_path.parent  # Adjust the number of parents as needed
    for _ in range(4):  # check up to 4 levels up
        if (potential_root / "pyproject.toml").exists():
            return potential_root.as_posix()
        potential_root = potential_root.parent
    return current_path.parent.as_posix()


def read_yaml_file(file):
    """Read a yaml file and return the content as a dictionary.

    Args:
        file: str, the file path
    Return:
        dict, the content of the yaml
    """
    try:
        with open(file, "r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)
    except FileNotFoundError:
        return None
    except yaml.YAMLError:
        return None
