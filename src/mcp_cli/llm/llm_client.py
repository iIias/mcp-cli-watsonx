# mcp_cli/llm/llm_client.py
from mcp_cli.llm.providers.base import BaseLLMClient

def get_llm_client(provider="openai", model="gpt-4o-mini", api_key=None) -> BaseLLMClient:
    if provider == "watsonx":
        # import
        from mcp_cli.llm.providers.watsonx_client import WatsonxLLMClient

        return WatsonxLLMClient(model=model)
    elif provider == "openai":

        # import
        from mcp_cli.llm.providers.openai_client import OpenAILLMClient

        # return the open ai client
        return OpenAILLMClient(model=model, api_key=api_key, api_base=api_base)

    elif provider == "ollama":
        # import
        from mcp_cli.llm.providers.ollama_client import OllamaLLMClient

        # return the ollama client
        return OllamaLLMClient(model=model)
    else:
        # unsupported provider
        raise ValueError(f"Unsupported provider: {provider}")
