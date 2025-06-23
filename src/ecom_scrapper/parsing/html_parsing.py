import re
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import tiktoken
from bs4 import BeautifulSoup, Tag

from ecom_scrapper.utils import get_logger


class HTMLContentFilter:
    """Intelligent HTML content filter for LLM processing."""

    def __init__(self, target_fields: List[str], max_tokens: int = 20000):
        self.target_fields = target_fields
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.logger = get_logger(__name__)
        # Build field-related keywords
        self.field_keywords = self._build_field_keywords()

    def _build_field_keywords(self) -> Set[str]:
        """Build a set of keywords related to target fields."""
        keywords = set()
        for field in self.target_fields:
            # Add the field name itself
            keywords.add(field.lower())
            # Add common variations
            keywords.add(field.lower().replace("_", "-"))
            keywords.add(field.lower().replace("-", "_"))
            # Add partial matches
            for word in field.lower().split("_"):
                if len(word) > 2:
                    keywords.add(word)
        return keywords

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    def remove_irrelevant_content(self, html_content: str) -> str:
        """Remove content that's unlikely to contain target fields."""
        # Remove script and style blocks completely
        html_content = re.sub(r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r"<style[^>]*>.*?</style>", "", html_content, flags=re.DOTALL | re.IGNORECASE)

        # Remove comments
        html_content = re.sub(r"<!--.*?-->", "", html_content, flags=re.DOTALL)

        # Remove common non-content elements
        irrelevant_patterns = [
            r"<meta[^>]*>",
            r"<link[^>]*>",
            r"<noscript[^>]*>.*?</noscript>",
            r"<iframe[^>]*>.*?</iframe>",
        ]

        for pattern in irrelevant_patterns:
            html_content = re.sub(pattern, "", html_content, flags=re.DOTALL | re.IGNORECASE)

        return html_content

    # pylint:disable=too-many-branches,too-many-statements,too-many-nested-blocks
    def extract_urls(self, html_content: str, base_url: Optional[str] = None) -> Dict[str, List[str]]:
        """Extract all URLs referenced in the HTML content.

        Args:
            html_content: The HTML content to parse
            base_url: Base URL to resolve relative URLs (optional)

        Returns:
            Dictionary categorizing URLs by type:
            {
                'links': [...],        # Links from <a> tags
                'images': [...],       # Images from <img> tags
                'scripts': [...],      # Scripts from <script> tags
                'stylesheets': [...],  # Stylesheets from <link> tags
                'forms': [...],        # Form actions from <form> tags
                'media': [...],        # Video/audio sources
                'iframes': [...],      # Iframe sources
                'other': [...]         # Other URL references
            }
        """
        soup = BeautifulSoup(html_content, "html.parser")
        urls = {
            "links": [],
            "images": [],
            "scripts": [],
            "stylesheets": [],
            "forms": [],
            "media": [],
            "iframes": [],
            "other": [],
        }

        def normalize_url(url: str) -> Optional[str]:
            """Normalize and resolve relative URLs."""
            if not url or url.startswith(("javascript:", "mailto:", "tel:", "#")):
                return None

            url = url.strip()
            if base_url and not url.startswith(("http://", "https://", "//")):
                try:
                    url = urljoin(base_url, url)
                except Exception:  # pylint: disable=broad-exception-caught
                    return None

            return url

        def is_valid_url(url: Optional[str]) -> bool:
            """Check if URL is valid and not a data URI or fragment."""
            if not url:
                return False

            # Skip data URIs, fragments, and javascript
            if url.startswith(("data:", "#", "javascript:", "mailto:", "tel:")):
                return False

            try:
                parsed = urlparse(url)
                return bool(parsed.netloc or parsed.path)
            except Exception:  # pylint: disable=broad-exception-caught
                return False

        # Extract links from <a> tags
        for link in soup.find_all("a", href=True):
            if isinstance(link, Tag):
                href = link.get("href")
                if href:
                    url = normalize_url(str(href))
                    if url and is_valid_url(url) and url not in urls["links"]:
                        urls["links"].append(url)

        # Extract images from <img> tags
        for img in soup.find_all("img", src=True):
            if isinstance(img, Tag):
                src = img.get("src")
                if src:
                    url = normalize_url(str(src))
                    if url and is_valid_url(url) and url not in urls["images"]:
                        urls["images"].append(url)

        # Extract scripts from <script> tags
        for script in soup.find_all("script", src=True):
            if isinstance(script, Tag):
                src = script.get("src")
                if src:
                    url = normalize_url(str(src))
                    if url and is_valid_url(url) and url not in urls["scripts"]:
                        urls["scripts"].append(url)

        # Extract stylesheets from <link> tags
        for link in soup.find_all("link", href=True):
            if isinstance(link, Tag):
                rel = link.get("rel")
                if rel:
                    if isinstance(rel, list):
                        rel_str = " ".join(rel).lower()
                    else:
                        rel_str = str(rel).lower()

                    if "stylesheet" in rel_str or "icon" in rel_str:
                        href = link.get("href")
                        if href:
                            url = normalize_url(str(href))
                            if url and is_valid_url(url) and url not in urls["stylesheets"]:
                                urls["stylesheets"].append(url)

        # Extract form actions from <form> tags
        for form in soup.find_all("form", action=True):
            if isinstance(form, Tag):
                action = form.get("action")
                if action:
                    url = normalize_url(str(action))
                    if url and is_valid_url(url) and url not in urls["forms"]:
                        urls["forms"].append(url)

        # Extract media sources from <video>, <audio>, <source> tags
        for media in soup.find_all(["video", "audio"], src=True):
            if isinstance(media, Tag):
                src = media.get("src")
                if src:
                    url = normalize_url(str(src))
                    if url and is_valid_url(url) and url not in urls["media"]:
                        urls["media"].append(url)

        for source in soup.find_all("source", src=True):
            if isinstance(source, Tag):
                src = source.get("src")
                if src:
                    url = normalize_url(str(src))
                    if url and is_valid_url(url) and url not in urls["media"]:
                        urls["media"].append(url)

        # Extract iframe sources
        for iframe in soup.find_all("iframe", src=True):
            if isinstance(iframe, Tag):
                src = iframe.get("src")
                if src:
                    url = normalize_url(str(src))
                    if url and is_valid_url(url) and url not in urls["iframes"]:
                        urls["iframes"].append(url)

        # Extract URLs from CSS background-image properties
        style_pattern = r'background-image\s*:\s*url\(["\']?([^"\')\s]+)["\']?\)'
        for element in soup.find_all(style=True):
            if isinstance(element, Tag):
                style_content = element.get("style")
                if style_content:
                    matches = re.findall(style_pattern, str(style_content), re.IGNORECASE)
                    for match in matches:
                        url = normalize_url(match)
                        if url and is_valid_url(url) and url not in urls["other"]:
                            urls["other"].append(url)

        # Extract URLs from srcset attributes (responsive images)
        for img in soup.find_all(["img", "source"], srcset=True):
            if isinstance(img, Tag):
                srcset = img.get("srcset")
                if srcset:
                    # Parse srcset format: "url1 1x, url2 2x" or "url1 100w, url2 200w"
                    srcset_urls = re.findall(r"([^\s,]+)(?:\s+[0-9.]+[wx])?", str(srcset))
                    for srcset_url in srcset_urls:
                        url = normalize_url(srcset_url)
                        if url and is_valid_url(url) and url not in urls["images"]:
                            urls["images"].append(url)

        # Extract meta refresh URLs
        for meta in soup.find_all("meta", attrs={"http-equiv": re.compile(r"refresh", re.I)}):
            if isinstance(meta, Tag):
                content = meta.get("content")
                if content:
                    # Format: "5; url=http://example.com"
                    match = re.search(r"url=([^;]+)", str(content), re.IGNORECASE)
                    if match:
                        url = normalize_url(match.group(1).strip())
                        if url and is_valid_url(url) and url not in urls["other"]:
                            urls["other"].append(url)

        # Extract canonical URLs
        for link in soup.find_all("link", rel="canonical", href=True):
            if isinstance(link, Tag):
                href = link.get("href")
                if href:
                    url = normalize_url(str(href))
                    if url and is_valid_url(url) and url not in urls["other"]:
                        urls["other"].append(url)

        # Log extraction results
        total_urls = sum(len(url_list) for url_list in urls.values())
        self.logger.info(
            f"Extracted {total_urls} URLs: "
            f"links={len(urls['links'])}, "
            f"images={len(urls['images'])}, "
            f"scripts={len(urls['scripts'])}, "
            f"stylesheets={len(urls['stylesheets'])}, "
            f"forms={len(urls['forms'])}, "
            f"media={len(urls['media'])}, "
            f"iframes={len(urls['iframes'])}, "
            f"other={len(urls['other'])}"
        )

        return urls

    def get_all_urls(self, html_content: str, base_url: Optional[str] = None) -> List[str]:
        """Get all unique URLs from HTML content as a flat list.

        Args:
            html_content: The HTML content to parse
            base_url: Base URL to resolve relative URLs (optional)

        Returns:
            List of all unique URLs found in the HTML
        """
        categorized_urls = self.extract_urls(html_content, base_url)
        all_urls = set()

        for url_list in categorized_urls.values():
            all_urls.update(url_list)

        return sorted(list(all_urls))

    def get_product_related_urls(self, html_content: str, base_url: Optional[str] = None) -> List[str]:
        """Extract URLs that are likely related to products (for e-commerce scraping).

        Args:
            html_content: The HTML content to parse
            base_url: Base URL to resolve relative URLs (optional)

        Returns:
            List of URLs that likely point to products or product-related content
        """
        categorized_urls = self.extract_urls(html_content, base_url)
        product_urls = []

        # Product-related keywords to look for in URLs
        product_keywords = [
            "product",
            "item",
            "detail",
            "p/",
            "/p/",
            "catalog",
            "shop",
            "buy",
            "purchase",
            "goods",
            "merchandise",
            "listing",
        ]

        # Check links for product-related patterns
        for url in categorized_urls["links"]:
            url_lower = url.lower()
            if any(keyword in url_lower for keyword in product_keywords):
                product_urls.append(url)

        # Also include product images as they often indicate product pages
        product_urls.extend(categorized_urls["images"])

        return list(set(product_urls))  # Remove duplicates

    def extract_relevant_sections(self, html_content: str) -> str:
        """Extract only sections likely to contain target fields."""
        soup = BeautifulSoup(html_content, "html.parser")
        relevant_elements = []

        # Priority 1: Elements with attributes matching field names
        for field in self.target_fields:
            field_lower = field.lower()

            # Find by ID, class, name, data attributes
            selectors = [
                f'[id*="{field_lower}"]',
                f'[class*="{field_lower}"]',
                f'[name*="{field_lower}"]',
                f'[data-*="{field_lower}"]',
            ]

            for selector in selectors:
                elements = soup.select(selector)
                relevant_elements.extend(elements)

        # Priority 2: Form elements (likely to contain data fields)
        form_elements = soup.find_all(["form", "input", "select", "textarea", "label"])
        relevant_elements.extend(form_elements)

        # Priority 3: Main content areas
        content_selectors = [
            "main",
            '[role="main"]',
            ".main-content",
            "#main-content",
            ".content",
            "#content",
            "article",
            ".article",
        ]

        for selector in content_selectors:
            elements = soup.select(selector)
            relevant_elements.extend(elements)

        # Priority 4: Elements containing field keywords in text
        all_elements = soup.find_all(text=True)
        for element in all_elements:
            if element.parent and any(keyword in str(element).lower() for keyword in self.field_keywords):
                # Include parent element and its context
                parent = element.parent
                for _ in range(2):  # Go up 2 levels for context
                    if parent.parent:
                        parent = parent.parent
                relevant_elements.append(parent)

        # Remove duplicates while preserving order
        unique_elements = []
        seen = set()
        for elem in relevant_elements:
            if elem and id(elem) not in seen:
                unique_elements.append(elem)
                seen.add(id(elem))

        # Convert back to HTML
        if unique_elements:
            filtered_html = "\n".join(str(elem) for elem in unique_elements)
        else:
            # Fallback: keep body content only
            body = soup.find("body")
            filtered_html = str(body) if body else html_content

        return filtered_html

    def compress_attributes(self, html_content: str) -> str:
        """Remove unnecessary HTML attributes to reduce token count."""
        soup = BeautifulSoup(html_content, "html.parser")

        # Attributes to keep (potentially relevant for field extraction)
        keep_attributes = {
            "id",
            "class",
            "name",
            "for",
            "value",
            "placeholder",
            "title",
            "alt",
            "aria-label",
            "data-field",
            "data-name",
        }

        # Add field-related data attributes
        for field in self.target_fields:
            keep_attributes.add(f"data-{field.lower()}")

        for element in soup.find_all():
            if isinstance(element, Tag) and element.attrs:
                # Keep only relevant attributes
                new_attrs = {}
                for attr, value in element.attrs.items():
                    if (
                        attr in keep_attributes
                        or attr.startswith("data-")
                        and any(field.lower() in attr.lower() for field in self.target_fields)
                    ):
                        new_attrs[attr] = value
                element.attrs = new_attrs

        return str(soup)

    def intelligent_truncate(self, html_content: str) -> str:
        """Intelligently truncate content while preserving structure."""
        current_tokens = self.count_tokens(html_content)
        if current_tokens <= self.max_tokens:
            return html_content

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove elements in order of decreasing importance
        removal_priorities = [
            # Lowest priority: decorative elements
            ["svg", 'img[alt=""]', ".icon", ".decoration"],
            # Navigation elements
            ["nav", ".navigation", ".menu", ".breadcrumb"],
            # Footer content
            ["footer", ".footer"],
            # Sidebar content
            [".sidebar", ".aside", "aside"],
        ]

        for priority_group in removal_priorities:
            if self.count_tokens(str(soup)) <= self.max_tokens:
                break

            for selector in priority_group:
                elements = soup.select(selector)
                for elem in elements:
                    # Don't remove if contains field keywords
                    elem_text = elem.get_text().lower()
                    if not any(keyword in elem_text for keyword in self.field_keywords):
                        elem.decompose()

                        # Check if we're under limit
                        if self.count_tokens(str(soup)) <= self.max_tokens:
                            return str(soup)

        # Final fallback: truncate by character count
        html_str = str(soup)
        if self.count_tokens(html_str) > self.max_tokens:
            # Rough approximation: 4 characters per token
            target_chars = self.max_tokens * 4
            html_str = html_str[:target_chars] + "..."

        return html_str

    def filter_html(self, html_content: str) -> str:
        """Apply all filtering rules to HTML content."""
        self.logger.info(f"Original content: {self.count_tokens(html_content)} tokens")

        # Step 1: Remove irrelevant content
        html_content = self.remove_irrelevant_content(html_content)
        self.logger.info(f"After removing irrelevant content: {self.count_tokens(html_content)} tokens")

        # Step 2: Extract relevant sections
        html_content = self.extract_relevant_sections(html_content)
        self.logger.info(f"After extracting relevant sections: {self.count_tokens(html_content)} tokens")

        # Step 3: Compress attributes
        html_content = self.compress_attributes(html_content)
        self.logger.info(f"After compressing attributes: {self.count_tokens(html_content)} tokens")

        # Step 4: Intelligent truncation if still too large
        html_content = self.intelligent_truncate(html_content)
        self.logger.info(f"Final content: {self.count_tokens(html_content)} tokens")

        return html_content
