from typing import Union

from .base import LLM_REGISTRY, BaseChatModel, ModelServiceError
from .oai import TextChatAtOAI
from .qwenvl_oai import QwenVLChatAtOAI


def get_chat_model(cfg: Union[dict, str] = 'qwen2.5-vl-7b-instruct') -> BaseChatModel:
    """The interface of instantiating LLM objects.

    Args:
        cfg: The LLM configuration, one example is:
          cfg = {
              # Use the model service provided by DashScope:
              'model': 'qwen-max',
              'model_server': 'dashscope',

              # Use your own model service compatible with OpenAI API:
              # 'model': 'Qwen',
              # 'model_server': 'http://127.0.0.1:7905/v1',

              # (Optional) LLM hyper-parameters:
              'generate_cfg': {
                  'top_p': 0.8,
                  'max_input_tokens': 6500,
                  'max_retries': 10,
              }
          }

    Returns:
        LLM object.
    """

    if 'model_type' in cfg:
        model_type = cfg['model_type']
        if model_type in LLM_REGISTRY:
            return LLM_REGISTRY[model_type](cfg)
        else:
            raise ValueError(f'Please set model_type from {str(LLM_REGISTRY.keys())}')

    raise ValueError(f'Invalid model cfg: {cfg}')


__all__ = [
    'BaseChatModel',
    'TextChatAtOAI',
    'QwenVLChatAtOAI',
    'get_chat_model',
    'ModelServiceError',
]
