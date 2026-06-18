from review.agent import BasicAgent, ParallelDocQA
from review.configs import CONFMAP
from typing import List

llm_config = CONFMAP.get('model_config')

# 非流式、流式
def chitchat(messages: List[dict]):
    """
    Args:
        messages:
            example1：input text only
                [
                {"role": "user", "content": "你好！"}
                ]
            example2：input text and image
                [
                    {"role": "user", "content": [{"image": "image_url"}, {"text": "请解释一下这张图片"}]}
                ]
            example3：input text and images
                [
                    {"role": "user", "content": [{"image": "image_url1"}, {"image": image_url2}, {"text": "请解释一下这两张图片"}]}
                ]

    return: iterator
            [{"role": "assistant", "content": "delta "}]
    """
    chat_obj = BasicAgent(llm_config)
    chat_response = chat_obj.run(messages)
    return chat_response

# 非流式、流式
def review_chat(messages: List[dict]):
    """
    Args:
        messages:
                [
                {"role": "user", "content": [{"file": "file1_url"}, {"file": "file2_url"}, {"file": "file3_url"}]}
                ]
    return: iterator
            [{"role": "assistant", "content": [{"text": "review result "}, {"image": "image_path"}, {"task": "verify task"}]}]
    """
    doc_obj = ParallelDocQA(llm_config)
    chat_response = doc_obj.run(messages)
    return chat_response


