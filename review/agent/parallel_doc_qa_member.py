import copy
from typing import Dict, Iterator, List, Optional, Union

from .agent import Agent
from review.llm.base import BaseChatModel
from review.llm.schema import DEFAULT_SYSTEM_MESSAGE, ROLE, SYSTEM, USER, ContentItem, Message

SYSTEM_PROMPT_TEMPLATE_REVIEW_TABLE = """
你是一名专业的表格内容审核员，负责严格依据给定的审核点与规则，对表格内容进行一致性检查。请完全忽略图像中文字的颜色特征。红色、黑色、蓝色等所有颜色的文字都应被视为相同的黑色文字进行处理。
"""


SYSTEM_PROMPT_TEMPLATE = {
    'review_table': SYSTEM_PROMPT_TEMPLATE_REVIEW_TABLE,
}


class ParallelDocQAMember(Agent):

    def __init__(self,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 **kwargs):
        super().__init__(llm, system_message, **kwargs)

    def _run(self,
             messages: List[Message],
             task: str = '',
             prompt: str = '',
             **kwargs) -> Iterator[List[Message]]:
        messages = copy.deepcopy(messages)

        system_prompt = SYSTEM_PROMPT_TEMPLATE[task]
        messages.insert(0, Message(SYSTEM, system_prompt))
        assert messages[-1][ROLE] == USER, messages
        messages[-1].content.append(ContentItem(text=prompt))

        return self._call_llm(messages=messages, stream=False)
