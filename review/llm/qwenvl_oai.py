import copy
import logging
import os
from pprint import pformat
from typing import List

from .base import register_llm, ModelServiceError
from .oai import TextChatAtOAI
from .schema import ContentItem, Message
from review.utils.log import logger
from review.utils.utils import (encode_image_as_base64, encode_video_as_base64, save_url_to_local_work_dir)
from review.utils.settings import DEFAULT_WORKSPACE

@register_llm('qwenvl_oai')
class QwenVLChatAtOAI(TextChatAtOAI):

    @property
    def support_multimodal_input(self) -> bool:
        return True

    @staticmethod
    def convert_messages_to_dicts(messages: List[Message]) -> List[dict]:
        new_messages = []
        for msg in messages:
            content = msg.content
            if isinstance(content, str):
                content = [ContentItem(text=content)]
            assert isinstance(content, list)

            new_content = []
            for item in content:
                t, v = item.get_type_and_value()
                if t == 'text' and v:
                    new_content.append({'type': 'text', 'text': v})
                if t == "image":
                    v = conv_multimodel_value(t, v)
                    new_content.append({'type': 'image_url', 'image_url': {'url': v}})


            new_msg = msg.model_dump()
            new_msg['content'] = new_content
            new_messages.append(new_msg)

        if logger.isEnabledFor(logging.DEBUG):
            lite_messages = copy.deepcopy(new_messages)
            for msg in lite_messages:
                for item in msg['content']:
                    if item.get('image_url', {}).get('url', '').startswith('data:'):
                        item['image_url']['url'] = item['image_url']['url'][:64] + '...'
                    if item.get('video_url', {}).get('url', '').startswith('data:'):
                        item['video_url']['url'] = item['video_url']['url'][:64] + '...'
                    if item.get('input_audio', {}).get('data', '').startswith('data:'):
                        item['input_audio']['data'] = item['input_audio']['data'][:64] + '...'

            logger.debug(f'LLM Input:\n{pformat(lite_messages, indent=2)}')

        return new_messages


def conv_multimodel_value(t, v):
    if os.path.exists(v):
        if t == 'image':
            v = encode_image_as_base64(v, max_short_side_length=1080)
        elif t == 'video':
            v = encode_video_as_base64(v)
        else:
            raise TypeError

    elif v.startswith(('http://', 'https://')):
        tmp_dir = os.path.join(DEFAULT_WORKSPACE, 'temporary')
        os.makedirs(tmp_dir, exist_ok=True)
        v = save_url_to_local_work_dir(v, tmp_dir)
        if t == 'image':
            v = encode_image_as_base64(v, max_short_side_length=1080)
        elif t == 'video':
            v = encode_video_as_base64(v)
        else:
            raise TypeError
    elif v.startswith('data:'):
        pass
    else:
        raise ModelServiceError(f'file "{v}" does not exist.')

    return v
