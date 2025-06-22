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
import urllib3
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ecom_scrapper.utils import get_logger, get_project_root, read_yaml_file


class SimpleCrawler:
    """Intelligent web crawler with proxy support and anti-detection features.

    This crawler provides functionality to:
    - Crawl web pages with respect to robots.txt
    - Use proxy rotation for anonymity
    - Parse sitemaps for URL discovery
    - Analyze robots.txt files
    - Handle rate limiting and retries
    - Save crawled content to files

    Attributes:
        config: Configuration loaded from system_config.yaml
        output_dir: Directory where output files are saved
        logger: Logger instance for this class
        use_proxy: Whether to use proxy rotation
        proxies_file: Path to file containing proxy list
        session: HTTP session with connection pooling
    """

    def __init__(
        self, 
        output_dir: str, 
        use_proxy: bool = False, 
        proxies_file: Optional[str] = None, 
        verify_ssl: bool = True,
        connect_timeout: int = 10,
        read_timeout: int = 30
    ) -> None:
        """Starts a crawler with the proxy config.

        Args:
            proxies_file: str, path to the file containing the proxies
            output_dir: str, path to the directory where the output will be saved
            use_proxy: bool, whether to use proxies or not
            verify_ssl: bool, whether to verify SSL certificates (default: True)
            connect_timeout: int, timeout for establishing connection (default: 10 seconds)
            read_timeout: int, timeout for reading response (default: 30 seconds)
        """
        config_path = Path(get_project_root()).joinpath("config", "system_config.yaml")
        self.config = read_yaml_file(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # setup logger
        self.logger = get_logger(__name__)
        self.use_proxy = use_proxy
        self.verify_ssl = verify_ssl
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout

        if not verify_ssl:
            self.logger.warning("SSL certificate verification is disabled. This may pose security risks.")
            # Disable SSL warnings to avoid cluttering logs
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        if self.use_proxy:
            self.proxies_file = proxies_file
            # validate the proxies file
            self._validate_proxies_file()

        # Setup session with connection pooling and retry strategy
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a requests session with connection pooling and retry strategy."""
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1,
        )

        # Configure HTTP adapter with retry strategy
        adapter = HTTPAdapter(pool_connections=1, pool_maxsize=5, max_retries=retry_strategy)

        # Mount adapter for both HTTP and HTTPS
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

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

    def review_visited_urls(self, file_name: str | Path, urls: List[str]) -> List[str]:
        """Filter out URLs that have already been visited.

        Args:
            file_name: Path to the file containing visited URLs
            urls: List of URLs to check

        Returns:
            List of URLs that haven't been visited yet
        """
        if not Path(file_name).exists():
            return urls

        visited_urls = set()
        try:
            with open(file_name, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and " --> " in line:
                        visited_url = line.split(" --> ")[0].strip()
                        visited_urls.add(visited_url)
        except (IOError, OSError) as e:
            self.logger.warning(f"Could not read visited URLs file {file_name}: {e}")
            return urls

        # Filter out already visited URLs
        urls_to_crawl = []
        for url in urls:
            if url in visited_urls:
                self.logger.info(f"URL {url} already visited, skipping.")
            else:
                urls_to_crawl.append(url)

        self.logger.info(f"Filtered {len(urls) - len(urls_to_crawl)} already visited URLs")
        return urls_to_crawl

    def crawl_pages(self, urls: List[str], max_pages: int = 50, crawl_file: Optional[str] = None) -> str:
        """Crawls the given URLs and saves the results to a file.

        Args:
            urls: List of URLs to crawl
            max_pages: Maximum number of pages to crawl
            crawl_file: Optional path to output file

        Returns:
            Path to the output file

        Raises:
            ValueError: If urls list is empty or max_pages is invalid
            IOError: If unable to write to output file
        """
        if not urls:
            raise ValueError("URLs list cannot be empty")
        if max_pages <= 0:
            raise ValueError("max_pages must be greater than 0")

        if not crawl_file:
            timestamp = int(time.time())
            output_file = self.output_dir / f"crawl_iter_{timestamp}.txt"
        else:
            output_file = Path(crawl_file)

        try:
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            rp = self._get_robot_parser(urls[0])
            realistic_user_agents = self.config.get("realistic_user_agents", [])

            if not realistic_user_agents:
                self.logger.warning("No realistic user agents found in config, using default")
                dummy_user_agent = "*"
            else:
                dummy_user_agent = random.choice(realistic_user_agents)

            visited, count = set(), 0
            urls_to_crawl = self.review_visited_urls(output_file, urls)

            self.logger.info(f"Starting to crawl {len(urls_to_crawl)} URLs (max: {max_pages})")

            with open(output_file, "w", encoding="utf-8") as f:
                for url in urls_to_crawl:
                    if count >= max_pages:
                        self.logger.info(f"Reached maximum pages limit ({max_pages})")
                        break

                    try:
                        if not rp.can_fetch(dummy_user_agent, url):
                            f.write(f"{url} --> forbidden by robotstxt for {dummy_user_agent}\n")
                            self.logger.debug(f"URL {url} forbidden by robots.txt")
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

                                try:
                                    with open(file_name, "w", encoding="utf-8") as html_file:
                                        html_file.write(response.text)
                                except IOError as e:
                                    self.logger.error(f"Failed to save HTML for {url}: {e}")
                                    f.write(f"{url} --> {response.status_code} --> ERROR_SAVING_HTML\n")
                        elif error:
                            f.write(f"{url} --> ERROR: {error}\n")
                        else:
                            f.write(f"{url} --> Unreachable\n")

                    except Exception as e:
                        self.logger.error(f"Unexpected error processing {url}: {e}")
                        f.write(f"{url} --> UNEXPECTED_ERROR: {str(e)}\n")

            self.logger.info(f"Crawling completed. Processed {count} pages. Results saved to {output_file}")
            return str(output_file)

        except IOError as e:
            self.logger.error(f"Failed to write to output file {output_file}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during crawling: {e}")
            raise

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
            timeout = (self.connect_timeout, self.read_timeout)
            response = self.session.get(robots_url, headers=headers, timeout=timeout, verify=self.verify_ssl)
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
        """Get the sitemap URLs from the given sitemap URL.

        Args:
            url: Base URL to search for sitemaps
            request: Whether to make HTTP requests (unused parameter)
            path_site: Optional path to local sitemap file

        Returns:
            List of URLs found in the sitemap
        """
        if not request and path_site:
            try:
                if not Path(path_site).exists():
                    self.logger.warning(f"Sitemap file {path_site} does not exist")
                    return []

                with open(path_site, "r", encoding="utf-8") as f:
                    sitemap_data = BeautifulSoup(f.read(), features="lxml")
                return [loc.text.strip() for loc in sitemap_data.find_all("loc")]
            except (IOError, OSError) as e:
                self.logger.error(f"Failed to read sitemap file {path_site}: {e}")
                return []
            except Exception as e:
                self.logger.error(f"Failed to parse sitemap file {path_site}: {e}")
                return []

        url = url.rstrip("/")  # Ensure URL ends with no trailing slash
        sitemap_urls = [
            f"{url}/sitemap.xml",
            f"{url}/sitemap_index.xml",
            f"{url}/sitemap/sitemap.xml",
        ]

        sitemap_data = None
        for sitemap_url in sitemap_urls:
            try:
                response, error = self.make_realistic_request(sitemap_url)
                if response and response.status_code == 200:
                    sitemap_data = BeautifulSoup(response.content, "lxml")
                    self.logger.info(f"Successfully fetched sitemap from {sitemap_url}")
                    break
                else:
                    self.logger.debug(f"Failed to fetch sitemap from {sitemap_url}: {error}")
            except Exception as e:
                self.logger.error(f"Error processing sitemap {sitemap_url}: {e}")
                continue

        if sitemap_data is None:
            self.logger.warning(f"No valid sitemap found for {url}")
            return []

        try:
            urls = [loc.text.strip() for loc in sitemap_data.find_all("loc")]
            self.logger.info(f"Found {len(urls)} URLs in sitemap")
            return urls
        except Exception as e:
            self.logger.error(f"Failed to extract URLs from sitemap: {e}")
            return []

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
            # Use tuple for separate connect and read timeouts
            timeout = (self.connect_timeout, self.read_timeout)
            response = self.session.get(url, headers=headers, proxies=proxies, timeout=timeout, verify=self.verify_ssl)
            response.raise_for_status()
            self.logger.info(f"Successfully fetched {url} with status code {response.status_code}")
        except requests.RequestException as e:
            if "403" in str(e) or "Access Denied" in str(e):
                self.logger.info("Access Denied, trying with a different User-agent")
                if trying_number >= max_retries:
                    self.logger.error(f"Failed to fetch {url} after {max_retries} attempts.")
                    return None, str(e)
                return self.make_realistic_request(url, max_retries=max_retries, trying=trying_number)
            elif "timeout" in str(e).lower() or "ReadTimeoutError" in str(e):
                self.logger.warning(f"Timeout error for {url} (attempt {trying_number}/{max_retries}): {e}")
                if trying_number >= max_retries:
                    self.logger.error(f"Failed to fetch {url} after {max_retries} timeout attempts.")
                    return None, str(e)
                # Add exponential backoff for timeout retries
                backoff_time = 2 ** trying_number
                self.logger.info(f"Retrying {url} in {backoff_time} seconds due to timeout...")
                time.sleep(backoff_time)
                return self.make_realistic_request(url, max_retries=max_retries, trying=trying_number)
            else:
                self.logger.error(f"Error fetching {url}: {e}")
                return None, str(e)
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
    argparser.add_argument(
        "--no-ssl-verify",
        dest="no_ssl_verify",
        action="store_true",
        help="Disable SSL certificate verification (use with caution)",
        default=False,
    )
    argparser.add_argument(
        "--connect-timeout",
        dest="connect_timeout",
        type=int,
        help="Connection timeout in seconds (default: 10)",
        default=10,
    )
    argparser.add_argument(
        "--read-timeout",
        dest="read_timeout",
        type=int,
        help="Read timeout in seconds (default: 30)",
        default=30,
    )

    args = argparser.parse_args()
    USE_PROXY = True
    if args.proxies_file is None:
        USE_PROXY = False
    crawler = SimpleCrawler(
        use_proxy=USE_PROXY,
        proxies_file=args.proxies_file,
        output_dir=args.output_dir,
        verify_ssl=not args.no_ssl_verify,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
    )
    response, error = crawler.make_realistic_request(args.url)
    if response:
        print(f"response is not none, status code: {response.status_code}")
        print(response.text)
    else:
        print(error)
    # URLS = crawler._get_sitemap_urls(args.url, path_site="data/results_scrapper/sitemap.xml")
    # print(URLS)
    # result = crawler.analyze_robots_txt_with_headers(args.url)
    # result = crawler.analyze_sitemap(url=args.url)
    # result = crawler.crawl_pages(
    #     url=args.url,
    # )
    # print(result)
