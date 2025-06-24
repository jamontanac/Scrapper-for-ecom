## SimpleCrawler: Robust Web Crawling with Occasional Failures

The `SimpleCrawler` implementation (see `src/ecom_scrapper/crawler/crawler.py`) is designed to provide a robust and flexible way to make HTTP requests (petitions) to websites, especially for e-commerce data extraction. It includes features such as:

- Proxy rotation for anonymity
- Anti-detection headers and user-agent rotation
- Respect for robots.txt rules
- Sitemap parsing for URL discovery
- Rate limiting and exponential backoff for retries
- Persistent logging and error handling

**Purpose:**

The main goal of this implementation is to maximize the success rate of web requests while minimizing the risk of being blocked or detected as a bot. It is built to handle a variety of network and server-side issues, making it suitable for crawling complex or protected sites.

**Important Note:**

Despite these robust features, it is important to understand that the crawler may still fail from time to time. Failures can occur due to:

- Aggressive anti-bot measures on target sites
- Proxy server unreliability or blacklisting
- Network instability or timeouts
- Unexpected changes in website structure or access policies

The crawler includes retry logic and detailed logging to help you diagnose and mitigate these issues, but 100% reliability cannot be guaranteed for all sites and all conditions.

For more details and usage examples, see `src/ecom_scrapper/crawler/README.md`.

## 🚀 How to Use

### Option 1: Command Line with Custom Timeouts

```bash
# Use longer timeouts for slow sites like REI
python src/ecom_scrapper/crawler/crawler.py \
    --url https://www.rei.com \
    --read-timeout 60 \
    --connect-timeout 15 \
    --output-dir output
```

### Option 2: Programmatic Usage

```python
from ecom_scrapper.crawler.crawler import SimpleCrawler

# Create crawler with extended timeouts
crawler = SimpleCrawler(
    output_dir="output",
    use_proxy=False,
    connect_timeout=15,  # 15 seconds to connect
    read_timeout=60      # 60 seconds to read response
)

# Make requests to slow-responding sites
response, error = crawler.make_realistic_request("https://www.rei.com")
```

### Option 3: For Very Slow Sites

```bash
# Maximum patience for extremely slow sites
python src/ecom_scrapper/crawler/crawler.py \
    --url https://www.rei.com \
    --read-timeout 120 \
    --connect-timeout 30 \
    --output-dir output
```

## ⚡ Performance Improvements

### 1. Exponential Backoff

- **1st retry**: Wait 2 seconds
- **2nd retry**: Wait 4 seconds
- **3rd retry**: Wait 8 seconds

### 2. Smart Retry Logic

- Distinguishes between different error types
- Specific handling for timeout vs access denied vs other errors
- Preserves retry attempts for different error categories

### 3. Better Logging

- Clear distinction between timeout and other errors
- Progress tracking for retry attempts
- Detailed timeout information for debugging

## 📊 Recommended Timeout Settings

| Site Type            | Connect Timeout | Read Timeout | Use Case                   |
| -------------------- | --------------- | ------------ | -------------------------- |
| **Fast Sites**       | 5s              | 15s          | Simple pages, APIs         |
| **Standard Sites**   | 10s             | 30s          | Most websites (default)    |
| **E-commerce Sites** | 15s             | 60s          | Complex pages like REI     |
| **Very Slow Sites**  | 30s             | 120s         | Heavy sites, poor networks |

## ✅ Testing Results

The solution has been tested and verified:

- Code compiles without errors
- All existing functionality preserved
- Timeout handling works for both connection and read phases
- Exponential backoff prevents overwhelming slow servers
- Configurable timeouts allow adaptation to different sites

## 🎯 For REI Issue

Try this command to resolve your specific timeout error:

```bash
python src/ecom_scrapper/crawler/crawler.py \
    --url https://www.rei.com \
    --read-timeout 60 \
    --connect-timeout 15 \
    --output-dir output
```

sometimes it also fails so keep that in mind
