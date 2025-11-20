import backoff
import openai
from .pricing import OPENAI_MODELS, get_model_pricing
from .result import QueryResult
import logging

logger = logging.getLogger(__name__)


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.warning(
            f"OpenAI - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
    max_tries=3,
    max_value=3,
    on_backoff=backoff_handler,
)
def query_openai(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query OpenAI model."""
    logger.debug(f"Querying OpenAI model: {model}")
    logger.debug(f"System message length: {len(system_msg) if system_msg else 0}")
    logger.debug(f"User message length: {len(msg) if msg else 0}")
    logger.debug(f"Message history length: {len(msg_history)}")
    logger.debug(f"Output model: {output_model}")
    logger.debug(f"Additional kwargs: {list(kwargs.keys())}")
    
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    if output_model is None:
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_msg},
                    *new_msg_history,
                ],
                **kwargs,
            )
            logger.debug(f"Response received. Type: {type(response)}")
            logger.debug(f"Response attributes: {dir(response)}")
            if hasattr(response, 'output'):
                logger.debug(f"Response output length: {len(response.output) if response.output else 0}")
        except Exception as e:
            logger.error(f"Failed to create response: {type(e).__name__}: {str(e)}")
            logger.error(f"Model: {model}")
            logger.error(f"Client base URL: {getattr(client, 'base_url', 'unknown')}")
            raise
        
        try:
            content = response.output[0].content[0].text
            logger.debug(f"Successfully extracted content from response.output[0].content[0].text")
        except Exception as e1:
            logger.warning(f"Failed to extract content using standard method: {type(e1).__name__}: {str(e1)}")
            # Reasoning models - ResponseOutputMessage
            # Try different output indices
            try:
                if len(response.output) > 1:
                    content = response.output[1].content[0].text
                    logger.debug(f"Successfully extracted content from response.output[1].content[0].text")
                elif hasattr(response, 'output_text'):
                    content = response.output_text
                    logger.debug(f"Successfully extracted content from response.output_text")
                else:
                    logger.error(f"Response structure: {response}")
                    logger.error(f"Response output: {response.output if hasattr(response, 'output') else 'N/A'}")
                    raise ValueError(f"Unexpected response format: {response}")
            except Exception as e2:
                logger.error(f"All content extraction methods failed: {type(e2).__name__}: {str(e2)}")
                raise
        
        logger.debug(f"Content extracted successfully. Length: {len(content) if content else 0}")
        new_msg_history.append({"role": "assistant", "content": content})
    else:
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                *new_msg_history,
            ],
            text_format=output_model,
            **kwargs,
        )
        content = response.output_parsed
        new_content = ""
        for i in content:
            new_content += i[0] + ":" + i[1] + "\n"
        new_msg_history.append({"role": "assistant", "content": new_content})

    # 计算成本 - 使用辅助函数获取定价
    pricing = get_model_pricing(model, OPENAI_MODELS)
    input_cost = pricing["input_price"] * response.usage.input_tokens
    output_cost = pricing["output_price"] * response.usage.output_tokens
    
    if model not in OPENAI_MODELS:
        logger.debug(
            f"模型 '{model}' 不在定价表中，使用零价格。"
            f"Input tokens: {response.usage.input_tokens}, Output tokens: {response.usage.output_tokens}"
        )
    else:
        logger.debug(f"使用模型 {model} 的定价")
    
    logger.debug(f"Cost calculation: input=${input_cost:.6f}, output=${output_cost:.6f}, total=${input_cost + output_cost:.6f}")
    
    result = QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost=input_cost + output_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought="",
        model_posteriors=model_posteriors,
    )
    logger.debug(f"Query result created successfully. Total cost: ${result.cost:.6f}")
    return result
