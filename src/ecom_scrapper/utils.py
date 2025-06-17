import json
import logging
import os
import pathlib
import subprocess
from typing import List

import colorlog
import requests
import yaml


def get_updated_proxy_list(country_codes: List[str]) -> None:
    """Retrieves the updated proxy_list.

    this function is based on the repo of
    https://github.com/vakhov/fresh-proxy-list/tree/master

    Args:
        contry_code: str, The code of certain country

    """
    url = os.getenv("PROXY_URL", "https://vakhov.github.io/fresh-proxy-list/proxylist.json")

    proxy_path = pathlib.Path(get_project_root()).joinpath("data", "proxies", "proxylist.json")
    proxy_path.parent.mkdir(parents=True, exist_ok=True)

    # make request to get the updated file
    response = requests.get(url, timeout=3)
    response.raise_for_status()

    # create the jq filter
    countries_json = json.dumps(country_codes)
    jq_filter = """
            [.[] | select(
                (.country_code as $cc | $countries | index($cc)) 
                and 
                (.socks4 == "0") 
                and 
                (.socks5 == "0")
            )]
            """.strip()

    try:
        result = subprocess.run(
            ["jq", "--argjson", "countries", countries_json, jq_filter],
            input=response.text,
            capture_output=True,
            text=True,
            check=True,
        )
        with open(proxy_path, "w", encoding="utf-8") as f:
            f.write(result.stdout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"jq error: {e.stderr}") from e

    except FileNotFoundError:
        raise RuntimeError("jq command not found - install jq first") from None


def get_logger(logger_name):
    """Create and return logger.

    logger is created to log into the console with log level Info.
    the string format will difer if the logger is being created from a script or from a module.

    Args:
        logger_name: str, the name of the logger
    Returns:
        logging.Logger: the logger
    """
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
    return current_path.as_posix()


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
