from typing import Any, Tuple
import os
import anthropic
import openai
import instructor
from pathlib import Path
from dotenv import load_dotenv
from .models.pricing import (
    CLAUDE_MODELS,
    BEDROCK_MODELS,
    M,
    OPENAI_MODELS,
    DEEPSEEK_MODELS,
    GEMINI_MODELS,
    ensure_model_in_pricing,
)

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


def get_client_llm(model_name: str, structured_output: bool = False) -> Tuple[Any, str]:
    """获取给定模型名称的客户端和模型。
    
    默认所有模型都使用 OpenAI 接口。如果模型不在定价表中，会自动添加零价格。

    Args:
        model_name: 模型名称
        structured_output: 是否需要结构化输出

    Returns:
        客户端和模型名称的元组
    """
    # 确保模型在定价表中（如果不存在则添加零价格）
    ensure_model_in_pricing(model_name, OPENAI_MODELS)
    
    # 默认使用 OpenAI 客户端
    client = openai.OpenAI()
    if structured_output:
        client = instructor.from_openai(client, mode=instructor.Mode.TOOLS_STRICT)
    
    return client, model_name
