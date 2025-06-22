import argparse
import pathlib
import re
from typing import Any, Dict, List, Optional, Type, Union

import dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.language_models import BaseChatModel

# from openai.types.chat import ChatCompletionMessageParam
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

from ecom_scrapper.crawler.crawler import SimpleCrawler
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


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process and results."""

    fields_found: List[str] = Field(
        description="List of fields that were successfully found and extracted", default_factory=list
    )
    confidence_scores: Dict[str, float] = Field(
        description="Confidence scores (0.0-1.0) for each extracted field", default_factory=dict
    )
    extraction_method: Dict[str, str] = Field(description="Method used to extract each field", default_factory=dict)


class ExtractionAgent(BaseModel):
    """Class with the expected output for extraction."""

    extracted_data: Dict[str, str] = Field(
        description="Dictionary with the extracted data from the website. All values are strings, with empty strings for missing fields.",
        default_factory=dict,
    )
    extraction_metadata: ExtractionMetadata = Field(
        description="Comprehensive metadata about the extraction process including success rates, methods used, and confidence scores"
    )

    class config:
        extra = "forbid"


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


def format_dicts(data, indent_level=0) -> str:
    """Recursively format dictionaries and lists into a string with indentation."""
    final_str = ""
    if isinstance(data, dict):
        for key, value in data.items():
            final_str += "\t" * indent_level + key + "\n"
            final_str += format_dicts(value, indent_level + 1)
    elif isinstance(data, list):
        for item in data:
            final_str += format_dicts(item, indent_level)
    else:
        final_str += "\t" * indent_level + str(data) + "\n"
    return final_str


def format_and_send_messages_parsing(
    resource_path: str,
    model: str,
    fields_to_extract: Optional[List[str]] = None,
    config: Dict[str, Any] = load_config_parsing(),
):
    """Format the messages for the OpenAI chat completion request."""
    if not fields_to_extract:
        fields_to_extract = list(config["fields_to_extract"])
    # str_interests = format_dicts(fields_to_extract)
    str_fields = ", ".join(fields_to_extract)
    # str_resources = ""
    with open(resource_path, "r", encoding="utf-8") as f:
        content = f.read()
        content = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", content, flags=re.S)

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = splitter.split_text(content)
    llm_structured = create_structured_llm(select_llm_model(model), ExtractionAgent)
    main_prompt = config["system_prompt"]
    for chunk in chunks:
        user_prompt = config["user_prompt"].format(fields=str_fields, web_resource=chunk)
        messages_to_send = [
            create_generic_message(main_prompt, role="system", message_type="ai"),
            create_generic_message(user_prompt, role="user", message_type="user"),
        ]
        try:
            answer = llm_structured.invoke(messages_to_send)  # yield to process each chunk separately
        except ValidationError:
            answer = ExtractionAgent(extracted_data={}, extraction_metadata=ExtractionMetadata())
        yield answer

    # start the model


def format_and_send_messages_navigation(
    resources: List[str],
    model: str,
    interests: Optional[List[str]] = None,
    config: Dict[str, Any] = load_config_navigation(),
) -> NavigationAgent:
    """Format the messages for the OpenAI chat completion request."""
    if not interests:
        interests = list(config["interests"])
    # str_interests = format_dicts(interests)
    str_interests = ", ".join(interests)
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


def extract_fields_from_file(
    file_path: str,
    model: str = "gpt-4o",
):
    answers = format_and_send_messages_parsing(resource_path=file_path, model=model)
    merged_extracted_data = {}
    for answer in answers:
        merged_extracted_data.update(answer)

    return merged_extracted_data


def get_next_sites_from_sitemap(
    url: str,
    output_dir: str,
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
    )

    return answer


def get_next_sites_from_urls(
    urls: List[str],
    model: str = "gpt-4o",
) -> NavigationAgent:
    """Call the model to get the next sites from a list of URLs."""
    answer = format_and_send_messages_navigation(
        resources=urls,
        model=model,
    )
    return answer


def requests_urls(
    urls: List[str],
    output_dir: str,
    use_proxy: bool = False,
    proxy_file: Optional[str] = None,
):
    """this function is used to request the URLs and save the results in the output directory."""
    crawler = SimpleCrawler(output_dir=output_dir, proxies_file=proxy_file, use_proxy=use_proxy)
    output_file = crawler.crawl_pages(urls=urls)
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the navigation agent.")
    parser.add_argument("--url", type=str, required=True, help="The URL to navigate.")
    parser.add_argument(
        "--config-crawler-path", type=str, default="config/navigation_agent.yaml", help="Path to the config file."
    )
    parser.add_argument("--output-dir", type=str, default="data/results_scrapper", help="Directory to save results.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use.")
    parser.add_argument("--use-proxy", action="store_true", help="Use proxy for requests.", default=False)
    parser.add_argument("--proxy-file", type=str, default=None, help="Path to the proxy file.")

    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    # result =
    # result = get_next_sites_from_sitemap(
    #     url=args.url,
    #     output_dir=args.output_dir,
    #     model=args.model,
    #     use_proxy=args.use_proxy,
    #     proxy_file=args.proxy_file,
    # )
    result = extract_fields_from_file(file_path="data/results_scrapper/main_site.html")
    print(result)
#     messages = format_messages(resources, interests)
#     response = submit_request_to_openai(messages, model)
#     if response:
#         return response
#     else:
#         return "{}"
