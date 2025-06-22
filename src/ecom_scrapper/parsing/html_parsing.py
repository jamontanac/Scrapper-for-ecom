import re
from typing import List, Set

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
            # Large text blocks without field keywords
            ["p", "div"],
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
