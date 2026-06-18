import copy
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Union

from review.llm import get_chat_model
from review.llm.base import BaseChatModel
from review.llm.schema import CONTENT, DEFAULT_SYSTEM_MESSAGE, ROLE, SYSTEM, Message
from review.utils.utils import merge_generate_cfgs


class Agent(ABC):
    """A base class for Agent.
    """

    def __init__(self,
                 llm: Optional[Union[dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 **kwargs):
        """Initialization the agent.
        """
        if isinstance(llm, dict):
            self.llm = get_chat_model(llm)
        else:
            self.llm = llm
        self.extra_generate_cfg: dict = {}

        self.system_message = system_message

    def run_nonstream(self, messages: List[Union[Dict, Message]], **kwargs) -> Union[List[Message], List[Dict]]:
        *_, last_responses = self.run(messages, **kwargs)
        return last_responses

    def run(self, messages: List[Union[Dict, Message]],
            **kwargs) -> Union[Iterator[List[Message]], Iterator[List[Dict]]]:
        """Return one response generator based on the received messages.

        This method performs a uniform type conversion for the inputted messages,
        and calls the _run method to generate a reply.

        Args:
            messages: A list of messages.

        Yields:
            The response generator.
        """
        messages = copy.deepcopy(messages)
        _return_message_type = 'dict'
        new_messages = []
        # Only return dict when all input messages are dict
        if not messages:
            _return_message_type = 'message'
        for msg in messages:
            if isinstance(msg, dict):
                new_messages.append(Message(**msg))
            else:
                new_messages.append(msg)
                _return_message_type = 'message'

        if self.system_message:
            if new_messages[0][ROLE] != SYSTEM:
                # Add the system instruction to the agent
                new_messages.insert(0, Message(role=SYSTEM, content=self.system_message))
            else:
                # Already got system message in new_messages
                if isinstance(new_messages[0][CONTENT], str):
                    new_messages[0][CONTENT] = self.system_message + '\n\n' + new_messages[0][CONTENT]

        for rsp in self._run(messages=new_messages, **kwargs):
            # print(rsp)
            if _return_message_type == 'message':
                yield [Message(**x) if isinstance(x, dict) else x for x in rsp]
            else:
                yield [x.model_dump() if not isinstance(x, dict) else x for x in rsp]

    @abstractmethod
    def _run(self, messages: List[Message], **kwargs) -> Iterator[List[Message]]:
        """Return one response generator based on the received messages.
        """
        raise NotImplementedError

    def _call_llm(
        self,
        messages: List[Message],
        stream: bool = True,
        extra_generate_cfg: Optional[dict] = None,
    ) -> Iterator[List[Message]]:

        return self.llm.chat(messages=messages,
                             stream=stream,
                             extra_generate_cfg=merge_generate_cfgs(
                                 base_generate_cfg=self.extra_generate_cfg,
                                 new_generate_cfg=extra_generate_cfg,
                             ))


class BasicAgent(Agent):

    def _run(self, messages: List[Message], **kwargs) -> Iterator[List[Message]]:
        extra_generate_cfg = dict()
        if kwargs.get('seed') is not None:
            extra_generate_cfg['seed'] = kwargs['seed']
        return self._call_llm(messages, extra_generate_cfg=extra_generate_cfg)
