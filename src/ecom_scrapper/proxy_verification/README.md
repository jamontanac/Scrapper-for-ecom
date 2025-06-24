# Proxy Verification Module

This module provides robust tools for obtaining, filtering, and verifying proxy servers for use with the E-commerce Scrapper. While optional, this module significantly enhances the crawler's ability to bypass anti-scraping measures and distribute requests across multiple IP addresses.

## Overview

The Proxy Verification module serves as a critical support component that:

1. Retrieves up-to-date proxy server lists from reliable sources
2. Validates each proxy's functionality through parallel testing
3. Filters proxies by country or region when geographic targeting is needed
4. Saves verified proxies for later use by the crawler

## Key Features

### Proxy Retrieval

- Automatically fetches fresh proxy lists from external sources
- Supports both plain text and JSON formatted proxy lists
- Handles both anonymous and country-specific proxy sources

### Parallel Validation

- Uses thread pools to efficiently validate multiple proxies simultaneously
- Implements connection timeouts to quickly identify non-responsive proxies
- Tests each proxy against reliable endpoints to confirm functionality

### Filtering Options

- Country-based filtering to target specific geographic locations
- Format standardization to ensure compatibility with the crawler
- Result persistence to avoid re-validation of known good proxies

## Usage

### Basic Usage (Command Line)

```bash
# Verify all available proxies without country filtering
python -m ecom_scrapper.proxy_verification.proxy_verification

# Filter and verify proxies from specific countries
python -m ecom_scrapper.proxy_verification.proxy_verification --countries US CA MX
```

### Programmatic Usage

```python
from ecom_scrapper.proxy_verification.proxy_verification import get_filtrated_proxy_list

# Get proxies without country filtering
valid_proxies = get_filtrated_proxy_list(excecutors=8, save_file=True)

# Get proxies from specific countries
valid_proxies = get_filtrated_proxy_list(
    countries=["US", "CA", "MX"],
    excecutors=8,
    save_file=True
)
```

## Integration with Crawler

While optional, the Proxy Verification module provides significant benefits when used with the main crawler:

1. **Improved Success Rate**: Pre-verified proxies reduce request failures
2. **Anti-Ban Protection**: Rotating through valid proxies helps avoid IP-based blocking
3. **Geographic Testing**: Enables testing how websites appear from different countries
4. **Rate Limit Bypass**: Distributes requests across multiple IP addresses

## Configuration

The module can be configured with the following parameters:

- `countries`: List of country codes to filter proxies by (e.g., ["US", "CA"])
- `excecutors`: Number of parallel threads for validation (default: 4)
- `save_file`: Whether to save valid proxies to a file (default: False)

## Output Files

When `save_file=True`, the module generates one of the following files:

- `data/proxies/valid_proxieshttp.txt`: Plain text list of valid proxies
- `data/proxies/valid_proxies.json`: JSON-formatted list of valid proxies with metadata

## Notes

- Proxy validation is resource-intensive and may take several minutes for large proxy lists
- The quality of free proxy servers varies greatly; expect only 10-20% to be functional
- For production use, consider using paid proxy services for better reliability