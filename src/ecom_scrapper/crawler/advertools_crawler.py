import os
import random
import time
from pathlib import Path
from typing import List, Optional

import advertools as adv

from ecom_scrapper.utils import get_logger, read_yaml_file


class SmartCrawler:
    """Intelligent Crawler that uses advertools with proxies."""

    def __init__(self, proxies_file: str, output_dir: str, config_path: str) -> None:
        """Starts a crawler with the proxy config.

        Args:
            proxies_file: str, path to the file containing the proxies
            output_dir: str, path to the directory where the output will be saved
            config_path: str, path to the configuration file
        """
        self.proxies_file = proxies_file
        self.config = read_yaml_file(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # setup logger
        self.logger = get_logger(__name__)

        # validate the proxies file
        self._validate_proxies_file()

    def _validate_proxies_file(self) -> None:
        """Validates the proxies file."""
        if not os.path.exists(self.proxies_file):
            self.logger.warning(
                f"Proxies file {self.proxies_file} does not exist. Please provide a valid proxies file."
            )
            raise FileNotFoundError(f"Proxies file {self.proxies_file} does not exist.")
        if not os.path.isfile(self.proxies_file):
            raise ValueError(f"Proxies file {self.proxies_file} is not a file.")
        with open(self.proxies_file, "r", encoding="utf-8") as f:
            proxies = f.readlines()
        self.logger.info(f"Loaded {len(proxies)} proxies from {self.proxies_file}")

    def crawl_with_analysis(
        self,
        url: str,
        max_pages: int = 50,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> str:
        """Performs a crawl on the given URL and analyzes the results.

        Args:
            url: str, the URL to crawl
            max_pages: int, the maximum number of pages to crawl
            include_patterns: Optional[List[str]], patterns to include in the crawl
            exclude_patterns: Optional[List[str]], patterns to exclude from the crawl

        Returns:
            output_file: str, path to the file with the results
        """
        # Generate an unique name to the output file
        timestamp = int(time.time())
        output_file = self.output_dir / f"crawl_results_{timestamp}.jsonl"

        # read config
        custom_settings = {
            **self.config["custom_settings"],
            "ROTATING_PROXY_LIST_PATH": self.proxies_file,
            "CLOSESPIDER_PAGECOUNT": max_pages,
            "DOWNLOAD_DELAY": random.uniform(2, 4),  # Random delay for requests
        }
        if include_patterns:
            custom_settings["include_url_regex"] = "|".join(include_patterns)
        if exclude_patterns:
            custom_settings["exclude_url_regex"] = "|".join(exclude_patterns)

        try:
            # start the crawl
            self.logger.info(f"Starting crawl on {url} with max pages {max_pages}")

            adv.crawl(url_list=url, output_file=str(output_file), follow_links=True, custom_settings=custom_settings)

            self.logger.info(f"Crawl completed. Results saved to {output_file}")
            return str(output_file)
        except Exception as e:
            self.logger.error(f"Error during crawl: {e}")
            raise
