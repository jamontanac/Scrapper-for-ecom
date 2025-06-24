import argparse
import pathlib
import re
from typing import Any, Dict, List, Optional, Type, Union

import dotenv

# from openai.types.chat import ChatCompletionMessageParam
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from ecom_scrapper.crawler.crawler import SimpleCrawler
from ecom_scrapper.parsing.html_parsing import HTMLContentFilter
from ecom_scrapper.utils import get_logger, get_project_root, read_yaml_file

logger = get_logger(__name__)
dotenv.load_dotenv()


class NavigationAgent(BaseModel):
    """Class with the expected output."""

    urls: List[str] = Field(
        description="List of URLs found that are interesting for further navigation.",
        # json_schema_extra={"examples": ['urls:["https://website.com/camping","https://website.com/tents"]']},
    )
    rules: List[str] = Field(
        ...,
        description="List of rules that should be applied during the next extraction navigation.",
    )
    reasoning: List[str] = Field(
        ...,
        description="List of reasoning steps taken to arrive at the conclusions.",
    )


# Data Models
class ProductData(BaseModel):
    """Product information extracted from HTML."""

    name: Optional[str] = Field(default=None, description="The product name or title")
    price: Optional[str] = Field(default=None, description="The product price (e.g., '$29.95' or '$19.95 - $39.95')")
    id: Optional[str] = Field(default=None, description="Product ID or SKU if available")
    image_url: Optional[str] = Field(default=None, description="URL of the product image")
    description: Optional[str] = Field(default=None, description="Product description or key features")


class ProductList(BaseModel):
    """List of products extracted from HTML content."""

    products: List[ProductData] = Field(description="List of products found in the HTML content")


def create_extraction_chain(model_name: str, chunk: Any):
    """Create a chain to extract product information from HTML content using a specified LLM model."""
    config = load_config_parsing()
    prompt = config["system_prompt"]
    llm_model = select_llm_model(model_name)
    prompt_template = ChatPromptTemplate.from_template(template=prompt)
    parser = PydanticOutputParser(pydantic_object=ProductList)
    chain = prompt_template | llm_model | parser
    try:
        result = chain.invoke({"html_content": chunk[:3000], "format_instructions": parser.get_format_instructions()})
        return result
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error processing with LLM: {e}")
        return extract_from_text_content(chunk.page_content)


def select_llm_model(model_name) -> BaseChatModel:
    """Select the LLM model to use."""
    if "gpt" in model_name.lower():
        return ChatOpenAI(model=model_name, temperature=0.3)

    raise NotImplementedError(f"{model_name} is not supported yet")


def create_structured_llm(llm_model: BaseChatModel, data_structure: Type[BaseModel]) -> Runnable:
    """Creates a structured LLM model with a specific Pydantic output structure.

    Args:
        llm_model: Base LLM model to structure
        data_structure: Pydantic model class that defines the output structure

    Returns:
        Runnable: Structured LLM that outputs according to the provided schema

    Example:
        ```python
        class ResponseSchema(BaseModel):
            answer: str
            confidence: float

        base_llm = ChatAnthropic(model="claude-3-sonnet")
        structured_llm = create_structured_llm(base_llm, ResponseSchema)
        ```
    """
    return llm_model.with_structured_output(data_structure, method="function_calling")


def preprocess_html_file(html_file_path: str, max_chunk_size: int = 3000) -> List[Any]:
    """Preprocess HTML file to extract relevant product sections.

    Args:
        html_file_path: Path to HTML file
        max_chunk_size: Maximum size of each chunk

    Returns:
        List of document chunks containing product information
    """
    try:
        # Load HTML using LangChain's BSHTMLLoader
        loader = BSHTMLLoader(html_file_path)
        documents = loader.load()
        # Transform HTML to clean up unwanted elements
        bs_transformer = BeautifulSoupTransformer()

        cleaned_docs = bs_transformer.transform_documents(
            documents,
            tags_to_extract=["div", "span", "p", "h1", "h2", "h3", "h4", "img", "a", "li"],
            unwanted_tags=["script", "style", "nav", "footer", "header"],
            remove_lines=True,
        )

        # Split into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_documents(cleaned_docs)
        # Filter chunks that likely contain product information
        product_chunks = []
        product_indicators = ["$", "price", "buy", "add to cart", ".95", ".99", "product", "item"]

        for chunk in chunks:
            content_lower = chunk.page_content.lower()
            if any(indicator in content_lower for indicator in product_indicators):
                product_chunks.append(chunk)
        return product_chunks

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error preprocessing HTML file: {e}")
        return []


def extract_from_text_content(text_content: str) -> List[ProductData]:
    """Extract products from raw text content (fallback method).

    Args:
        text_content: Raw text content from HTML

    Returns:
        List of ProductData objects
    """
    # Simple pattern matching for the specific format in the provided HTML
    products = []

    # Pattern: - ProductName$Price
    pattern = r"-\s*(.+?)\$(\d+(?:\.\d{2})?(?:\s*-\s*\$\d+(?:\.\d{2})?)?)"
    matches = re.findall(pattern, text_content)

    for match in matches:
        name = match[0].strip()
        price = f"${match[1]}"

        # Clean product name (remove special indicators)
        name = re.sub(r"(Top Rated|Save \d+%)", "", name).strip()

        products.append(ProductData(name=name, price=price, id=None, image_url=None, description=None))

    return products


def create_generic_message(content: str, role: str, message_type="ai") -> Union[AIMessage, HumanMessage, ToolMessage]:
    """Create a generic message with the given content, role, and type."""
    if message_type == "tool":
        return ToolMessage(content=content, role=role)
    if message_type == "user":
        return HumanMessage(content=content, role=role)
    return AIMessage(content=content, role=role)


def load_config_navigation():
    """Load the configuration for the navigation agent from a YAML file."""
    config_file = pathlib.Path(get_project_root()).joinpath("config", "navigation_agent_config.yaml")
    config = read_yaml_file(config_file)
    return config


def load_config_parsing():
    """Load the configuration for the navigation agent from a YAML file."""
    config_file = pathlib.Path(get_project_root()).joinpath("config", "parsing_agent_config.yaml")
    config = read_yaml_file(config_file)
    return config


def format_and_send_messages_navigation(
    resources: List[str] | str,
    model: str,
    base_url: str,
    interests: Optional[List[str]] = None,
    config: Dict[str, Any] = load_config_navigation(),
) -> NavigationAgent:
    """Format the messages for the OpenAI chat completion request."""
    if not interests:
        interests = list(config["interests"])
    # str_interests = format_dicts(interests)
    str_interests = ", ".join(interests)
    if isinstance(resources, str):
        with open(resources, "r", encoding="utf-8") as f:
            html_content = f.read()
        filter_engine = HTMLContentFilter(interests)

        # Get categorized URLs

        all_urls = filter_engine.get_all_urls(html_content, base_url=base_url)

        str_resources = "\n".join(all_urls)
    else:
        str_resources = "\n".join(resources)
    main_prompt = config["system_prompt"]
    user_prompt = config["user_prompt"].format(interests=str_interests, web_resource=str_resources)
    messages_to_send = [
        create_generic_message(main_prompt, role="system", message_type="ai"),
        create_generic_message(user_prompt, role="user", message_type="user"),
    ]
    # start the model
    llm_structured = create_structured_llm(select_llm_model(model), NavigationAgent)

    answer = llm_structured.invoke(messages_to_send)
    return answer


def get_next_sites_from_sitemap(
    url: str,
    output_dir: str,
    base_url: str,
    model: str = "gpt-4o",
    use_proxy: bool = False,
    proxy_file: Optional[str] = None,
) -> NavigationAgent:
    """Call the model to get the next sites from a sitemap URL.

    Args:
        url (str): The URL to navigate.
        output_dir (str): Directory to save the results.
        model (str): The OpenAI model to use for the chat completion.
        use_proxy (bool): Whether to use a proxy for requests.
        proxy_file (Optional[str]): Path to the proxy file if using a proxy.

    Returns:
        str: The response from the OpenAI API.
    """
    crawler = SimpleCrawler(output_dir=output_dir, proxies_file=proxy_file, use_proxy=use_proxy)
    urls = crawler._get_sitemap_urls(url=url, path_site=f"{output_dir[:-1]}/sitemap.xml")
    answer = format_and_send_messages_navigation(
        resources=urls,
        model=model,
        base_url=base_url,
    )

    return answer


def get_next_sites_from_urls(
    urls: List[str],
    base_url: str,
    model: str = "gpt-4o",
) -> NavigationAgent:
    """Call the model to get the next sites from a list of URLs."""
    answer = format_and_send_messages_navigation(
        resources=urls,
        model=model,
        base_url=base_url,
    )
    return answer


def get_next_sites_from_file(
    html_file_path: str,
    base_url: str,
    model: str = "gpt-4o",
) -> NavigationAgent:
    """Get next sites from a file containing HTML content."""
    answer = format_and_send_messages_navigation(resources=html_file_path, base_url=base_url, model=model)
    return answer


def requests_urls(
    urls: List[str],
    output_dir: str,
    use_proxy: bool = False,
    proxy_file: Optional[str] = None,
):
    """This function is used to request the URLs and save the results in the output directory."""
    crawler = SimpleCrawler(output_dir=output_dir, proxies_file=proxy_file, use_proxy=use_proxy)
    output_file = crawler.crawl_pages(urls=urls)
    return output_file


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run the navigation agent.")
    arg_parser.add_argument("--url", type=str, required=True, help="The URL to navigate.")
    arg_parser.add_argument(
        "--config-crawler-path", type=str, default="config/navigation_agent.yaml", help="Path to the config file."
    )
    arg_parser.add_argument(
        "--output-dir", type=str, default="data/results_scrapper", help="Directory to save results."
    )
    arg_parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use.")
    arg_parser.add_argument("--use-proxy", action="store_true", help="Use proxy for requests.", default=False)
    arg_parser.add_argument("--proxy-file", type=str, default=None, help="Path to the proxy file.")

    args = arg_parser.parse_args()
    logger.info(f"Arguments: {args}")

    # chunks = preprocess_html_file(html_file_path=args.output_dir + "climbing.html", max_chunk_size=30000)
    # logger.info(f"Number of chunks extracted: {len(chunks)}")
    # if not chunks:
    #     logger.error("No chunks extracted from the HTML file. Please check the file content.")
    # result = []
    # for i, chunk in enumerate(chunks):
    #     logger.info(f"Processing chunk {i + 1}/{len(chunks)}")
    #     products = extract_from_text_content(chunk)
    #     result.extend(products)
    # print(result)

    # result = extract_fields_from_file(file_path="data/results_scrapper/main_site.html")
    # result = get_next_sites_from_file(html_file_path=args.output_dir + "main_site.html", base_url=args.url)
    # result = extract_fields_from_file(file_path="data/results_scrapper/climbing.html")
    # print(result)
#     messages = format_messages(resources, interests)
#     response = submit_request_to_openai(messages, model)
#     if response:
#         return response
#     else:
#         return "{}"
