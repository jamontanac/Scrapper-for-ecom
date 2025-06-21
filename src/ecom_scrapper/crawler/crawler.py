import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

# import advertools as adv
import pandas as pd
import requests
from bs4 import BeautifulSoup

from ecom_scrapper.utils import get_logger, get_project_root, read_yaml_file


class SimpleCrawler:
    """Intelligent Crawler that uses advertools with proxies."""

    def __init__(self, output_dir: str, use_proxy: bool = False, proxies_file: Optional[str] = None) -> None:
        """Starts a crawler with the proxy config.

        Args:
            proxies_file: str, path to the file containing the proxies
            output_dir: str, path to the directory where the output will be saved
            use_proxy: bool, whether to use proxies or not
        """
        config_path = Path(get_project_root()).joinpath("config", "system_config.yaml")
        self.config = read_yaml_file(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # setup logger
        self.logger = get_logger(__name__)
        self.use_proxy = use_proxy
        if self.use_proxy:
            self.proxies_file = proxies_file
            # validate the proxies file
            self._validate_proxies_file()

    def _create_random_name(self):
        animals = [
            "perro",
            "gato",
            "pajaro",
            "pez",
            "conejo",
            "tortuga",
            "caballo",
            "vaca",
            "oveja",
            "cerdo",
            "leon",
            "tigre",
            "elefante",
            "jirafa",
            "mono",
            "zorro",
            "lobo",
            "oso",
            "serpiente",
            "aguila",
        ]
        animal = random.choice(animals)
        date_str = datetime.now().strftime("%Y%m%d-%H%M")
        return f"Crawl-{animal}-{date_str}"

    def _validate_proxies_file(self) -> None:
        """Validates the proxies file."""
        if self.proxies_file is None:
            raise ValueError("Proxies file is not provided. Please provide a valid proxies file.")

        if not os.path.exists(self.proxies_file):
            self.logger.warning(
                f"Proxies file {self.proxies_file} does not exist. Please provide a valid proxies file."
            )
            raise FileNotFoundError(f"Proxies file {self.proxies_file} does not exist.")
        if not os.path.isfile(self.proxies_file):
            raise ValueError(f"Proxies file {self.proxies_file} is not a file.")
        with open(self.proxies_file, "r", encoding="utf-8") as f:
            self.proxies = f.readlines()
        self.logger.info(f"Loaded {len(self.proxies)} proxies from {self.proxies_file}")

    def crawl_pages(self, urls: List[str], max_pages: int = 50) -> str:
        """Crawls the given URLs and saves the results to a file."""
        timestamp = int(time.time())
        output_file = self.output_dir / f"crawl_iter_{timestamp}.txt"
        rp = self._get_robot_parser(urls[0])

        dummy_user_agent = random.choice(self.config.get("realistic_user_agents", []))
        visited, count = set(), 0
        with open(output_file, "w", encoding="utf-8") as f:
            for url in urls:
                if count >= max_pages:
                    break
                if not rp.can_fetch(dummy_user_agent, url):
                    f.write(f"{url} --> forbidden by robotstxt for {dummy_user_agent}")
                    continue
                response, error = self.make_realistic_request(url)
                if response:
                    count += 1
                    if url not in visited:
                        visited.add(url)
                        folder_file = self.output_dir / "crawl"
                        folder_file.mkdir(exist_ok=True)
                        file_name = folder_file / f"{self._create_random_name()}.html"
                        f.write(f"{url} --> {response.status_code} --> {file_name}\n")
                        with open(file_name, "w", encoding="utf-8") as html_file:
                            html_file.write(response.text)

                elif error != "":
                    f.write(f"{url} -> ERROR: {error} --> \n")
                else:
                    f.write(f"{url} --> Unreachable --> \n")
        return str(output_file)

    def _get_robot_parser(self, base_url: str) -> RobotFileParser:
        """Parse the robots.txt file for the given base URL."""
        rp = RobotFileParser()
        rp.set_url(urljoin(base_url, "/robots.txt"))
        rp.read()
        return rp

    def analyze_robots_txt_with_headers(self, url: str) -> Dict[str, Any]:
        """Analyzes the robots.txt file with custom anti-detection headers."""
        timestamp = int(time.time())
        output_file = self.output_dir / f"robots_analysis_{timestamp}.jl"
        robots_url = url.rstrip("/") + "/robots.txt"

        # Realistic user agents (same ones used in the crawler)

        realistic_user_agents = self.config.get("realistic_user_agents", [])

        headers = {
            **self.config["custom_settings"]["DEFAULT_REQUEST_HEADERS"],
            "User-Agent": random.choice(realistic_user_agents),
        }

        try:
            self.logger.info(f"Analyzing robots.txt for {robots_url} with anti-detection headers")

            # Make the request with custom headers
            response = requests.get(robots_url, headers=headers, timeout=30)
            response.raise_for_status()

            # Process the robots.txt content manually
            robots_df = self._parse_robots_content(response.text, robots_url)

            # Save to file if necessary
            if not robots_df.empty:
                robots_df.to_json(str(output_file), orient="records", lines=True, date_format="iso")

            # Generate analysis
            analysis = self._generate_robots_analysis(robots_df)

            self.logger.info("Robots.txt analysis completed successfully with custom headers.")
            return analysis

        except Exception as e:
            self.logger.error(f"Error during robots.txt analysis: {e}")
            raise

    def _get_sitemap_urls(self, url: str, request: bool = False, path_site: Optional[str] = None) -> List[str]:
        """Get the sitemap URLs from the given sitemap URL."""
        if not request and path_site:
            with open(path_site, "r", encoding="utf-8") as f:
                sitemap_data = BeautifulSoup(f.read(), "lxml")
            return [loc.text.strip() for loc in sitemap_data.find_all("loc")]

        url = url.rstrip("/")  # Ensure URL ends with no trailing slash
        sitemap_urls = [
            f"{url}/sitemap.xml",
            f"{url}/sitemap_index.xml",
            f"{url}/sitemap/sitemap.xml",
        ]

        sitemap_data = None
        for sitemap_url in sitemap_urls:
            response, _ = self.make_realistic_request(sitemap_url)
            if response:
                sitemap_data = BeautifulSoup(response.content, "lxml")
                break
            self.logger.info(f"Failed to fetch sitemap from {sitemap_url}, trying with other sitemap")

        if sitemap_data is None:
            return []
        return [loc.text.strip() for loc in sitemap_data.find_all("loc")]

    def _parse_robots_content(self, content: str, robots_url: str) -> pd.DataFrame:
        """Processes the robots.txt content and converts it to DataFrame."""
        lines = []

        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                if ":" in line:
                    parts = line.split(":", 1)
                    directive = parts[0].strip().lower()
                    content_part = parts[1].strip()
                    lines.append([directive, content_part])

        if not lines:
            return pd.DataFrame()

        df = pd.DataFrame(lines)
        df.columns = ["directive", "content"]
        df["robotstxt_url"] = robots_url
        df["download_date"] = pd.Timestamp.now(tz="UTC")

        return df

    def _generate_robots_analysis(self, robots_df: pd.DataFrame) -> Dict[str, Any]:
        """Generates the analysis of the robots.txt DataFrame."""
        if robots_df.empty:
            return {}

        # Extract user agents with explicit pandas operations
        user_agent_series = robots_df[robots_df["directive"] == "user-agent"]["content"]
        unique_user_agents = list(pd.Series(user_agent_series).drop_duplicates())

        analysis = {
            "total_rules": len(robots_df),
            "user_agents": unique_user_agents,
            "disallowed_paths": robots_df[robots_df["directive"] == "disallow"]["content"].tolist(),
            "allowed_paths": robots_df[robots_df["directive"] == "allow"]["content"].tolist(),
            "sitemaps": robots_df[robots_df["directive"] == "sitemap"]["content"].tolist(),
            "crawl_delay": robots_df[robots_df["directive"] == "crawl-delay"]["content"].tolist(),
        }

        return analysis

    def make_realistic_request(
        self, url: str, max_retries: int = 3, trying: int = 0
    ) -> Tuple[requests.Response | None, str]:
        """Performs a realistic HTTP request with anti-detection headers.

        Args:
            url (str): The URL to request.
            max_retries (int): Maximum number of retries for the request.
            trying (int): Current attempt number.

        Returns:
            Response or None: The HTTP response if successful, None otherwise.
        """
        url = url.strip("/")
        realistic_user_agents = self.config.get("realistic_user_agents", [])
        headers = {
            **self.config["custom_settings"]["DEFAULT_REQUEST_HEADERS"],
            "User-Agent": random.choice(realistic_user_agents),
        }
        proxy_to_use = random.choice(self.proxies) if self.use_proxy else None
        proxies = {"http": proxy_to_use} if proxy_to_use else None
        trying_number = trying + 1
        response = None
        try:
            response = requests.get(url, headers=headers, proxies=proxies, timeout=10)
            response.raise_for_status()
            self.logger.info(f"Successfully fetched {url} with status code {response.status_code}")
        except requests.RequestException as e:
            if "403" in str(e) or "Access Denied" in str(e):
                self.logger.info("Access Denied, trying with a different User-agent")
                if trying_number >= max_retries:
                    self.logger.error(f"Failed to fetch {url} after {max_retries} attempts.")
                    return None, str(e)
                self.make_realistic_request(url, max_retries=max_retries, trying=trying_number)
            else:
                self.logger.error(f"Error fetching {url}: {e}")
        finally:
            time.sleep(random.uniform(1, 3))  # Sleep to avoid overwhelming the server

        return response, ""


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Smart Crawler Script")
    argparser.add_argument(
        "--proxies-file", dest="proxies_file", type=str, help="Path to the proxies file", default=None
    )
    argparser.add_argument("--url", dest="url", type=str, help="URL to explore", required=True)
    argparser.add_argument(
        "--output-dir", dest="output_dir", type=str, help="Directory to save the output files", default="output"
    )
    argparser.add_argument(
        "--config-path", dest="config_path", type=str, help="Path to the configuration file", default="config.yaml"
    )

    args = argparser.parse_args()
    USE_PROXY = True
    if args.proxies_file is None:
        USE_PROXY = False
    crawler = SimpleCrawler(
        use_proxy=USE_PROXY,
        proxies_file=args.proxies_file,
        output_dir=args.output_dir,
    )
    # response, error = crawler.make_realistic_request(args.url)
    # if response:
    #     print(f"response is not none, status code: {response.status_code}")
    #     print(response.text)
    # else:
    #     print(error)
    # URLS = crawler._get_sitemap_urls(args.url, path_site="data/results_scrapper/sitemap.xml")
    # print(URLS)
    # result = crawler.analyze_robots_txt_with_headers(args.url)
    # result = crawler.analyze_sitemap(url=args.url)
    # result = crawler.crawl_pages(
    #     url=args.url,
    # )
    # print(result)
