import openai
import asyncio
import functools
import random
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

# Assume api_key.py contains API_CONFIGS dictionary
from api_key import API_CONFIGS


# --- 0. 请求/响应预处理 & 解析器函数 ---
# (These functions remain the same as in your original code)

def _preprocess_openai_o_series(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocessor for o1, o3, o4 models."""
    kwargs['messages'] = [
        {**msg, 'role': 'user'} if msg.get('role') == 'system' else msg
        for msg in kwargs.get('messages', [])
    ]
    kwargs['temperature'] = kwargs.get('temperature', 1.0)
    kwargs['max_tokens'] = kwargs.get('max_tokens', 8000)
    kwargs.pop('top_p', None)
    return kwargs

def _preprocess_thinking_stream(kwargs: Dict[str, Any], thinking_params: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocessor for models that support 'thinking' via extra_body."""
    kwargs['stream'] = True
    kwargs['extra_body'] = kwargs.get('extra_body', {})
    kwargs['extra_body'].update(thinking_params)
    return kwargs

async def default_response_parser(completion, is_stream: bool) -> str:
    """通用解析器：支持流式和非流式"""
    thinking_content, answer_content = "", ""
    if is_stream:
        async for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                thinking_content += delta.reasoning_content
            if hasattr(delta, "content") and delta.content:
                answer_content += delta.content
    else:
        message = completion.choices[0].message
        answer_content = message.content or ""

    if thinking_content:
        return f"<Thinking>{thinking_content}</Thinking>\n{answer_content}"
    return answer_content

async def deepseek_response_parser(completion, is_stream: bool) -> str:
    """DeepSeek 专用解析器 (assuming non-streaming for simplicity here)"""
    message = completion.choices[0].message
    answer_content = message.content or ""
    if hasattr(message, "reasoning_content"):
        thinking = message.reasoning_content
    else:
        thinking=None
    if thinking:
        return f"<Thinking>{thinking}</Thinking>\n{answer_content}"
    return answer_content


# --- 1. 集中配置 (REFACTORED) ---

PROVIDER_LOGIC = {
    "xiaoai": {
        "modes": {
            "default": {"request_preprocessor": None}
        },
        "response_parser": default_response_parser
    },
    "o1": {
        "modes": {
            "default": {"request_preprocessor": _preprocess_openai_o_series}
        },
        "response_parser": default_response_parser
    },
    "o3": {
        "modes": {
            "default": {"request_preprocessor": _preprocess_openai_o_series}
        },
        "response_parser": default_response_parser
    },
    "ali": {
        "modes": {
            "R": { # Reasoning Mode
                "request_preprocessor": functools.partial(
                    _preprocess_thinking_stream,
                    thinking_params={"enable_thinking": True, "thinking_budget": 16000}
                )
            },
            "default": { # Default Mode
                "request_preprocessor":  lambda kwargs: {**kwargs, "extra_body":{"enable_thinking": False}}
            }
        },
        "response_parser": default_response_parser
    },
    "claude": {
        "modes": {
            "R": { # Reasoning Mode
                 "request_preprocessor": functools.partial(
                    _preprocess_thinking_stream,
                    thinking_params={"thinking": {"type": "enabled", "budget_tokens": 4000}}
                )
            },
            "default": {
                "request_preprocessor": None
            }
        },
        "response_parser": default_response_parser
    },
    "ds": {
        "modes": {"default": {"request_preprocessor": None}},
        "response_parser": deepseek_response_parser
    },
    "minimax": {
        "modes": {
            "default": {"request_preprocessor": lambda kwargs: {**kwargs, "stream": True}}
        },
        "response_parser": default_response_parser
    }
}


# --- 2. 公共装饰器 ---
# (No changes here)

def async_adapter(func):
    """Decorator to run an async function in a sync context if no event loop is running."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                return func(*args, **kwargs)
        except RuntimeError:
            pass
        return asyncio.run(func(*args, **kwargs))
    return wrapper

def retry(max_retries=3, delay=1, exceptions=(Exception,)):
    """Decorator to retry a function upon catching specific exceptions."""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    print(f"Function {func.__name__} failed, retry {retries}/{max_retries}: {e}")
                    if retries >= max_retries:
                        print(f"Function {func.__name__} reached max retries, failing.")
                        raise
                    await asyncio.sleep(delay)
        return async_wrapper
    return decorator

# --- 3. ProviderConfig & Service (Partially REFACTORED) ---

@dataclass
class ProviderConfig:
    """Holds all configuration for a specific API provider."""
    base_url: str
    api_keys: List[str]
    # The preprocessor is no longer stored here, it's looked up dynamically.
    
class LLMApiService:
    """A unified service to interact with various LLM APIs."""

    def __init__(self):
        # This part is simplified as we don't need to store the preprocessor anymore
        self.provider_configs = {
            provider_name: ProviderConfig(
                base_url=config_data["base_url"],
                api_keys=config_data["api_keys"]
            )
            for provider_name, config_data in API_CONFIGS.items()
        }
        self.clients = {
            provider: [
                openai.AsyncOpenAI(base_url=config.base_url, api_key=key)
                for key in config.api_keys if key
            ]
            for provider, config in self.provider_configs.items()
        }

    def _get_client(self, provider: str) -> openai.AsyncOpenAI:
        """Gets a random client for the specified provider."""
        client_list = self.clients.get(provider)
        if not client_list:
            client_list = self.clients.get("default")
            if not client_list:
                raise ValueError(f"Provider '{provider}' or default provider not configured or has no valid API keys.")
        return random.choice(client_list)

    async def _process_response(self, provider: str, completion: Any, is_stream: bool) -> str:
        """根据 provider 调用对应的解析逻辑"""
        parser = PROVIDER_LOGIC.get(provider, {}).get("response_parser", default_response_parser)
        return await parser(completion, is_stream)

    @retry(max_retries=3, delay=2)
    async def chat(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        """Main method to get a chat completion. (REFACTORED)"""
        # --- NEW: Parse provider, model_name, and mode ---
        try:
            model_and_mode = model.split("@", 1)
            full_model_name = model_and_mode[0]
            mode = model_and_mode[1].upper() if len(model_and_mode) > 1 else "default"

            provider, model_name = full_model_name.split(":", 1)
        except ValueError:
            provider = "default"
            model_name = model
            mode = "default"
        # --- End of new parsing logic ---

        client = self._get_client(provider)

        request_kwargs = {"model": model_name, "messages": messages, **kwargs}
        
        # --- NEW: Dynamic lookup for preprocessor based on mode ---
        provider_logic = PROVIDER_LOGIC.get(provider, {})
        modes_config = provider_logic.get("modes", {})
        # Fallback to default mode if the specified mode doesn't exist
        mode_logic = modes_config.get(mode, modes_config.get("default", {}))
        
        preprocessor = mode_logic.get("request_preprocessor")
        if preprocessor:
            request_kwargs = preprocessor(request_kwargs)
        # --- End of dynamic lookup ---
        
        is_stream = request_kwargs.get('stream', False)
        completion = await client.chat.completions.create(**request_kwargs)

        return await self._process_response(provider, completion, is_stream)

    @retry(max_retries=3, delay=2)
    async def tool(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        """Main method to get a chat completion. (REFACTORED)"""
        # --- NEW: Parse provider, model_name, and mode ---
        try:
            model_and_mode = model.split("@", 1)
            full_model_name = model_and_mode[0]
            mode = model_and_mode[1].upper() if len(model_and_mode) > 1 else "default"

            provider, model_name = full_model_name.split(":", 1)
        except ValueError:
            provider = "default"
            model_name = model
            mode = "default"
        # --- End of new parsing logic ---

        client = self._get_client(provider)

        request_kwargs = {"model": model_name, "input": messages, **kwargs}
        
        # --- NEW: Dynamic lookup for preprocessor based on mode ---
        provider_logic = PROVIDER_LOGIC.get(provider, {})
        modes_config = provider_logic.get("modes", {})
        # Fallback to default mode if the specified mode doesn't exist
        mode_logic = modes_config.get(mode, modes_config.get("default", {}))
        
        preprocessor = mode_logic.get("request_preprocessor")
        if preprocessor:
            request_kwargs = preprocessor(request_kwargs)
        # --- End of dynamic lookup ---
        
        is_stream = request_kwargs.get('stream', False)
        completion = await client.responses.create(**request_kwargs)

        return  completion #await self._process_response(provider, completion, is_stream) #TODO 这里应该给tool写一个后处理
    
    # get_embedding method remains the same
    @retry(max_retries=3, delay=1)
    async def get_embedding(self, model: str, input: List[str], **kwargs) -> Any:
        """Gets text embeddings for a list of inputs."""
        try:
            provider, model_name = model.split(":", 1)
        except ValueError:
            provider = "default"
            model_name = model
        
        client = self._get_client(provider)
        response = await client.embeddings.create(model=model_name, input=input, **kwargs)
        return response.data


# --- 4. 全局实例 & 对外接口 ---
# (No changes here)
llm_service = LLMApiService()

@async_adapter
async def openai_response(model: str, **kwargs) -> str:
    """Public-facing interface for chat completions."""
    if 'messages' not in kwargs:
        raise ValueError("The 'messages' argument is required.")
    return await llm_service.chat(model=model, **kwargs)

@async_adapter
async def openai_tool(model: str, **kwargs) -> str:
    """Public-facing interface for chat completions."""
    if 'messages' not in kwargs:
        raise ValueError("The 'messages' argument is required.")
    return await llm_service.tool(model=model, **kwargs)


@async_adapter
async def get_embedding(model: str, input: List[str], **kwargs) -> any:
    """Public-facing interface for text embeddings."""
    return await llm_service.get_embedding(model=model, input=input, **kwargs)


# 示例
# response_qwen = await openai_response(model="ali:qwen3-235b-a22b@R", messages=messages)
