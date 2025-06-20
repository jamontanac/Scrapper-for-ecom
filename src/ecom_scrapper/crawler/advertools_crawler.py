import argparse
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import advertools as adv
import pandas as pd

from ecom_scrapper.utils import get_logger, read_yaml_file


class SmartCrawler:
    """Intelligent Crawler that uses advertools with proxies."""

    def __init__(
        self, output_dir: str, config_path: str, use_proxy: bool = False, proxies_file: Optional[str] = None
    ) -> None:
        """Starts a crawler with the proxy config.

        Args:
            proxies_file: str, path to the file containing the proxies
            output_dir: str, path to the directory where the output will be saved
            config_path: str, path to the configuration file
            use_proxy: bool, whether to use proxies or not
        """
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
        output_file = self.output_dir / f"crawl_results_{timestamp}.jl"
        realistic_user_agents = self.config.get("realistic_user_agents")
        custom_settings = {**self.config["custom_settings"],
            "USER_AGENT" : random.choice(realistic_user_agents) if realistic_user_agents else None,
            "MAX_PAGES" : max_pages,
            "DOWNLOAD_DELAY": random.uniform(5, 10)  # Random delay for requests,
                               }
        # read config
        if self.use_proxy:
            custom_settings = {
                **self.config["custom_settings"],
                "ROTATING_PROXY_LIST_PATH": self.proxies_file,
                "DOWNLOADER_MIDDLEWARES": {
                'rotating_proxies.middlewares.RotatingProxyMiddleware': 610,
                "rotating_proxies.middlewares.BanDetectionMiddleware": 620
            }
            }
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
        except Exception as e:  # pylint:disable=broad-exception-caught
            self.logger.error(f"Error during crawl: {e}")
            raise

    def analyze_robots_txt(self, url: str) -> Dict[str, Any]:
        """Analyzes the robots.txt file of the given URL.

        Args:
            url: str, the URL to analyze

        Returns:
            output_file: str, path to the file with the robots.txt analysis results
        """
        # Generate an unique name to the output file
        timestamp = int(time.time())
        output_file = self.output_dir / f"robots_analysis_{timestamp}.jl"
        url = url.rstrip("/") + "/robots.txt"
        try:
            self.logger.info(f"Analyzing robots.txt for {url}")
            robots_df = adv.robotstxt_to_df(robotstxt_url=url, output_file=str(output_file))
            if robots_df is None:
                return {}
            analysis = {
                "total_rules": len(robots_df),
                "users_agents": robots_df["user_agent"].unique().tolist(),
                "dissallowd_paths": robots_df[robots_df["directive"] == "Disallowed"]["content"].tolist(),
                "allowed_paths": robots_df[robots_df["directive"] == "Allowed"]["content"].tolist(),
                "sitemaps": robots_df[robots_df["directive"] == "Sitemap"]["content"].tolist(),
                "crawl_delay": robots_df[robots_df["directive"] == "Crawl-delay"]["content"].tolist(),
            }
            self.logger.info("Robots.txt analisis completed successfully.")
            return analysis

        except Exception as e:  # pylint:disable=broad-exception-caught
            self.logger.error(f"Error during robots.txt analysis: {e}")
            raise

    def analyze_sitemap(self, url: str) -> Dict[str, Any]:
        """Analyzes the sitemap of the given URL.

        Args:
            url: str, the URL to analyze the sitemap for

        Returns:
            Analysis results as a dictionary with the following keys:
            total_urls, url_types, last_modified, priority_distribution,
            change_frequency_distribution
        """
        try:
            # try to find it
            url = url.rstrip("/")  # Ensure URL ends with no trailing slash
            sitemap_urls = [
                f"{url}/sitemap.xml",
                f"{url}/sitemap_index.xml",
                f"{url}/sitemap/sitemap.xml",
            ]

            sitemap_data = None
            for sitemap_url in sitemap_urls:
                try:
                    self.logger.info(f"Trying to fetch sitemap from {sitemap_url}")
                    sitemap_data = adv.sitemap_to_df(sitemap_url)
                    if not sitemap_data.empty:
                        self.logger.info(f"Sitemap found at {sitemap_url}")
                        break
                except Exception as e:  # pylint:disable=broad-exception-caught
                    self.logger.warning(f"Failed to fetch sitemap from {sitemap_url}: {e}")
                    continue
            if sitemap_data is None:
                return {"Error": "sitemap not found"}
            analysis = {
                "total_urls": len(sitemap_data),
                "url_types": (sitemap_data.get("loc", pd.Series()) or pd.Series())
                .str.extract(r"\.(\w+)$")[0]
                .value_counts()
                .to_dict()
                if "loc" in sitemap_data.columns
                else {},
                "last_modified": (sitemap_data.get("lastmod", pd.Series()) or pd.Series()).describe().to_dict()
                if "lastmod" in sitemap_data.columns
                else {},
                "priority_distribution": (sitemap_data.get("priority", pd.Series()) or pd.Series()).describe().to_dict()
                if "priority" in sitemap_data.columns
                else {},
                "change_frequency_distribution": (sitemap_data.get("changefreq", pd.Series()) or pd.Series())
                .value_counts()
                .to_dict()
                if "changefreq" in sitemap_data.columns
                else {},
            }
            self.logger.info("Sitemap analysis completed successfully.")
            return analysis
        except Exception as e:  # pylint:disable=broad-exception-caught
            self.logger.warning(f"No se pudo analizar sitemap: {str(e)}")
            return {"error": str(e)}

    def load_crawl_data(self, crawl_file: str) -> pd.DataFrame:
        """Loads crawl data from JSONL file.

        Args:
            crawl_file: Path to the crawl file

        Returns:
            DataFrame with the crawl data
        """
        try:
            df = pd.read_json(crawl_file, lines=True)
            self.logger.info(f"Datos del crawl cargados: {len(df)} páginas")
            return df
        except Exception as e:  # pylint:disable=broad-exception-caught
            self.logger.error(f"Error cargando datos del crawl: {str(e)}")
            raise

    def get_crawl_summary(self, crawl_file: str) -> Dict[str, Any]:
        """Generates a summary of the performed crawl.

        Args:
            crawl_file: Path to the crawl file

        Returns:
            Summary with crawl statistics
        """
        df = self.load_crawl_data(crawl_file)

        summary = {
            "total_pages": len(df),
            "successful_requests": len(df[df["status"] == 200]),
            "failed_requests": len(df[df["status"] != 200]),
            "unique_domains": df["domain"].nunique() if "domain" in df.columns else 0,
            "content_types": df["resp_headers_content-type"].value_counts().head().to_dict()
            if "resp_headers_content-type" in df.columns
            else {},
            "status_codes": df["status"].value_counts().to_dict(),
            "proxy_usage": df.filter(regex="proxy").columns.tolist(),
            "avg_response_time": df["download_latency"].mean() if "download_latency" in df.columns else 0,
            "total_size_mb": df["size"].sum() / (1024 * 1024) if "size" in df.columns else 0,
        }

        return summary


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

    crawler = SmartCrawler(
        use_proxy=USE_PROXY, proxies_file=args.proxies_file, output_dir=args.output_dir, config_path=args.config_path
    )
    # result = crawler.analyze_robots_txt(args.url)
    result = crawler.analyze_sitemap(url=args.url)
    print(result)
