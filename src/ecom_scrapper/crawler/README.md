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
