# src/llm/providers/watsonx_client.py
import os
import json
import uuid
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

# base
from mcp_cli.llm.providers.base import BaseLLMClient

# utils
from mcp_cli.llm.tools_handler import parse_tool_response

# Load environment variables
load_dotenv()

class WatsonxLLMClient(BaseLLMClient):
    def __init__(self, model: str = "ibm/granite-3-8b-instruct"):
        self.model = model

        self.project_id = os.getenv("WATSONX_PROJECT_ID")
        api_key = os.getenv("WATSONX_API_KEY")
        endpoint = os.getenv("WATSONX_ENDPOINT_URL")

        if not all([self.project_id, api_key, endpoint]):
            raise ValueError("Missing Watsonx credentials in .env")

        credentials = Credentials(url=endpoint, api_key=api_key)
        self.client = ModelInference(
            model_id=self.model,
            credentials=credentials,
            project_id=self.project_id,
        )

    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        try:

            # Call chat() with tool support
            response = self.client.achat(
                messages=messages,
                tools=tools,
                tool_choice_option="auto"
            )

            message = response.get("choices", [{}])[0].get("message", {})
            tool_calls = message.get("tool_calls", [])
            content = message.get("content", "")

            # Convert tool_calls to standardized format
            parsed_tool_calls = []
            for tool in tool_calls:
                tool_call_id = tool.get("id", f"call_{uuid.uuid4().hex[:8]}")
                function = tool.get("function", {})
                name = function.get("name")
                arguments = function.get("arguments", {})

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass  # Leave as string if parsing fails

                parsed_tool_calls.append({
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                })

                print(f"Tool Calls: {parsed_tool_calls}")

            return {
                "response": "" if parsed_tool_calls else content,
                "tool_calls": parsed_tool_calls,
            }

        except Exception as e:
            logging.exception("Watsonx API Error")
            raise ValueError(f"Watsonx API Error: {e}")