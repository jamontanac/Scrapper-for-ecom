import argparse
import json
import pathlib
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Union

import requests

from ecom_scrapper.utils import (
    get_logger,
    get_project_root,
    get_updated_proxy_dict,
    get_updated_proxy_list,
)

logger = get_logger(__name__)


def get_filtrated_proxy_list(
    countries: List[str] | None = None, excecutors: int = 4, save_file=False
) -> Union[List[Dict[str, str]], List[str]]:
    """Get a filtered list of proxies based on the provided countries.

    Args:
        countries (list[str]): List of country codes to filter proxies by. Defualt  None
        excecutors (int): Number of threads to use for checking proxies.
        save_file (bool) Default False: Whether to save the valid proxies to a file.

    Returns:
        Union[List[Dict[str, str]], List[str]]: List of valid proxies, either as
        dictionaries with 'ip' and 'port' keys or as strings in the format 'ip:port'.
    """
    if isinstance(countries, List):
        get_updated_proxy_dict(country_codes=countries)
        proxy_file_path = pathlib.Path(get_project_root()).joinpath("data", "proxies", "proxylist.json")
        # Load the proxy list from the JSON file
        with open(proxy_file_path, "r", encoding="utf-8") as file:
            proxies = json.load(file)
    else:
        get_updated_proxy_list()
        proxy_file_path = pathlib.Path(get_project_root()).joinpath("data", "proxies", "proxylisthttp.txt")
        with open(proxy_file_path, "r", encoding="utf-8") as f:
            proxies = f.readlines()

    # Generate the queue to validate each element
    q = queue.Queue()
    for proxy in proxies:
        if isinstance(proxy, dict):
            q.put(proxy)
        else:
            q.put(proxy.strip())

    valid_proxies = check_proxies_parallel_executor(q, max_workers=excecutors)
    logger.info(f"Found {len(valid_proxies)} valid proxies out of {len(proxies)} total proxies.")
    if save_file:
        if isinstance(valid_proxies[0], str):
            new_proxy_file_path = pathlib.Path(get_project_root()).joinpath("data", "proxies", "valid_proxieshttp.txt")
            with open(new_proxy_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(valid_proxies))
        else:
            new_proxy_file_path = pathlib.Path(get_project_root()).joinpath("data", "proxies", "valid_proxies.json")
            with open(new_proxy_file_path, "w", encoding="utf-8") as f:
                json.dump(valid_proxies, f, indent=4)
        logger.info(f"Valid proxies saved to {new_proxy_file_path}")
    return valid_proxies


def check_proxies_parallel_executor(proxy_queue: queue.Queue, max_workers: int = 4):
    """Check proxies in parallel using ThreadPoolExecutor.

    Args:
        proxy_queue (queue.Queue): Queue containing proxy information
        max_workers (int): Maximum number of worker threads

    Returns:
        List[str]: List of valid proxy URLs
    """
    # Convert queue to list for ThreadPoolExecutor
    proxy_list = []
    while not proxy_queue.empty():
        proxy_list.append(proxy_queue.get())

    valid_proxies = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_proxy = {executor.submit(check_single_proxy, proxy_info): proxy_info for proxy_info in proxy_list}

        # Collect results as they complete
        for future in as_completed(future_to_proxy):
            result = future.result()
            if result:
                valid_proxies.append(result)

    return valid_proxies


def check_single_proxy(proxy_info: Union[Dict[str, str], str]):
    """Check if a single proxy is valid by making a request to a test URL.

    Args:
        proxy_info (Union[Dict[str, str], str]): Proxy information, either as a
        dictionary with 'ip' and 'port' keys or as a string in the format 'ip:port'.

    Returns:
        Union[Dict[str, str], str, None]: Returns the proxy information if valid,
        otherwise returns None.
    """
    if isinstance(proxy_info, dict):
        proxy = f"{proxy_info['ip']}:{proxy_info['port']}"
    else:
        proxy = proxy_info
    proxies = {"http": proxy}
    test_url = "http://ipinfo.io/json"
    try:
        response = requests.get(test_url, proxies=proxies, timeout=5)
        if response.status_code == 200:
            return proxy_info
    except requests.RequestException:
        # print(f"Proxy {proxy} is not valid.")
        # logger.info(f"Proxy {proxy} is not valid.")
        return None


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Proxy Verification Script")
    argparser.add_argument("--countries", dest="countries", nargs="+", type=str, default=None, required=False)
    args = argparser.parse_args()
    if args.countries is not None:
        logger.info(f"Starting proxy verification with countries: {args.countries}")
    else:
        logger.info("Starting proxy verification without country filtering.")
    get_filtrated_proxy_list(countries=args.countries, excecutors=4, save_file=True)
