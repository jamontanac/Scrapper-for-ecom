# E-commerce Parsing Module

This module provides intelligent parsing and navigation capabilities for e-commerce websites, working in conjunction with the crawler module to extract structured product information from web pages.

## Overview

The parsing module uses a combination of HTML parsing techniques and LLM-based extraction to identify and extract product information from e-commerce websites. It supports both structured navigation through sitemaps and exploratory navigation through HTML content analysis.

## Key Components

### Navigation Agent

The Navigation Agent allows for intelligent crawling of e-commerce sites by:

- Analyzing sitemaps to identify product-rich pages
- Parsing HTML content to discover relevant navigation paths
- Prioritizing URLs based on likelihood of containing product information
- Generating navigation rules that improve extraction efficiency

### HTML Content Parsing

The HTML parsing component:

- Preprocesses HTML to remove irrelevant content
- Extracts and normalizes URLs from various HTML elements
- Identifies product-related sections within HTML documents
- Provides utilities for token management and content filtering

### Product Data Extraction

The module continuously validates if it can extract the specified fields defined in the `ProductData` class:

- Product name
- Price
- Product ID
- Image URL
- Description

## Usage Patterns

### Sitemap-Based Navigation

```python
# Navigate using a sitemap
navigation_result = get_next_sites_from_sitemap(
    url="https://example.com",
    output_dir="data/results/",
    base_url="https://example.com",
    model="gpt-4o"
)
```

### HTML-Based Navigation

```python
# Navigate based on HTML content
navigation_result = get_next_sites_from_file(
    html_file_path="data/results/page.html",
    base_url="https://example.com",
    model="gpt-4o"
)
```

### Direct URL List Navigation

```python
# Navigate from a list of URLs
navigation_result = get_next_sites_from_urls(
    urls=["https://example.com/products", "https://example.com/categories"],
    base_url="https://example.com",
    model="gpt-4o"
)
```

## Field Extraction Validation

The module continuously checks if it can extract the fields specified in the `ProductData` class:

1. It preprocesses HTML to identify product-containing sections
2. It attempts to extract structured product data from these sections
3. If structured extraction fails, it falls back to text-based extraction methods
4. It validates extracted data against the expected schema
5. Missing fields are noted in the extraction logs

## Integration with Crawler

This module works closely with the `SimpleCrawler` to:

1. Request pages discovered through navigation
2. Process the HTML content retrieved by the crawler
3. Extract relevant product information
4. Identify new URLs for further crawling

## Limitations

- Field extraction accuracy depends on the structure of the target website
- LLM-based extraction may sometimes miss fields or extract incorrect values
- Some websites with complex JavaScript rendering may not be fully parsed
- Rate limiting and anti-bot measures may affect crawling efficiency
