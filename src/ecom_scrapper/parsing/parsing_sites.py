import argparse
import pathlib
from typing import Any, Dict, List, Optional, Type, Union

import dotenv
from langchain_core.language_models import BaseChatModel

# from openai.types.chat import ChatCompletionMessageParam
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ecom_scrapper.crawler.crawler import SimpleCrawler
from ecom_scrapper.utils import get_logger, get_project_root, read_yaml_file

logger = get_logger(__name__)
dotenv.load_dotenv()


class navigation_agent_output(BaseModel):
    """Class with the expected output."""

    urls: List[str] = Field(
        description="List of URLs found that are interesting for further navigation.",
        json_schema_extra={"examples": ['urls:["https://website.com/camping","https://website.com/tents"]']},
    )
    rules: List[str] = Field(
        ...,
        description="List of rules that should be applied during the next extraction navigation.",
    )


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

    return llm_model.with_structured_output(data_structure)


def create_generic_message(content: str, role: str, message_type="ai") -> Union[AIMessage, HumanMessage, ToolMessage]:
    """Create a generic message with the given content, role, and type."""
    if message_type == "tool":
        return ToolMessage(content=content, role=role)
    if message_type == "user":
        return HumanMessage(content=content, role=role)
    return AIMessage(content=content, role=role)


def load_config():
    """Load the configuration for the navigation agent from a YAML file."""
    config_file = pathlib.Path(get_project_root()).joinpath("config", "navigation_agent.yaml")
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


def format_and_send_messages(
    resources: List[str],
    interests: Optional[List[str]] = None,
    model: str = "gtp-4o",
    config: Dict[str, Any] = load_config(),
) -> Dict[str, Any]:
    """Format the messages for the OpenAI chat completion request."""
    main_prompt = config["system_prompt"]
    if not interests:
        interests = list(config["interests"].keys())
    # str_interests = format_dicts(interests)
    str_interests = ", ".join(interests)
    str_resources = "\n".join(resources)
    user_prompt = config["user_prompt"].format(interests=str_interests, web_resource=str_resources)
    messages_to_send = [
        create_generic_message(main_prompt, role="system", message_type="ai"),
        create_generic_message(user_prompt, role="user", message_type="user"),
    ]
    # start the model
    llm_structured = create_structured_llm(select_llm_model(model), navigation_agent_output)

    answer = llm_structured.invoke(messages_to_send)
    return answer


# def submit_request_to_openai(messages: List[ChatCompletionMessageParam], model: str = "gpt-3.5-turbo"):
#     response = openai.chat.completions.create(model=model, messages=messages, temperature=0.3, max_tokens=2500)
#     return response.choices[0].message.content


def run_navigation_agent(
    url: str,
    config_crawler_path: str,
    output_dir: str,
    model: str = "gpt-4o",
    use_proxy: bool = False,
    proxy_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the navigation agent with the provided resources and interests.

    Args:
        resources (List[str]): List of resources to navigate.
        interests (Optional[List[str]]): List of interests to consider.
        model (str): The OpenAI model to use for the chat completion.

    Returns:
        str: The response from the OpenAI API.
    """
    crawler = SimpleCrawler(
        output_dir=output_dir, config_path=config_crawler_path, proxies_file=proxy_file, use_proxy=use_proxy
    )
    urls = crawler._get_sitemap_urls(url=url, path_site="data/results_scrapper/sitemap.xml")
    answer = format_and_send_messages(
        resources=urls,
        model=model,
    )
    return answer


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
    result = run_navigation_agent(
        url=args.url,
        config_crawler_path=args.config_path,
        output_dir=args.output_dir,
        model=args.model,
        use_proxy=args.use_proxy,
        proxy_file=args.proxy_file,
    )
    print(result)
#     messages = format_messages(resources, interests)
#     response = submit_request_to_openai(messages, model)
#     if response:
#         return response
#     else:
#         return "{}"
