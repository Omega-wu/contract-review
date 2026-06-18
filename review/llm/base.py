import copy
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List,  Optional, Tuple, Union

from .schema import DEFAULT_SYSTEM_MESSAGE, Message, SYSTEM
from review.utils.log import logger

from review.utils.utils import  merge_generate_cfgs

LLM_REGISTRY = {}
DEFAULT_MAX_INPUT_TOKENS = 8000

def register_llm(model_type):

    def decorator(cls):
        LLM_REGISTRY[model_type] = cls
        return cls

    return decorator


class ModelServiceError(Exception):

    def __init__(self,
                 exception: Optional[Exception] = None,
                 code: Optional[str] = None,
                 message: Optional[str] = None,
                 extra: Optional[dict] = None):
        if exception is not None:
            super().__init__(exception)
        else:
            super().__init__(f'\nError code: {code}. Error message: {message}')
        self.exception = exception
        self.code = code
        self.message = message
        self.extra = extra


class BaseChatModel(ABC):
    """The base class of LLM"""

    @property
    def support_multimodal_input(self) -> bool:
        # Does the model support multimodal input natively? It affects how we preprocess the input.
        return False

    @property
    def support_multimodal_output(self) -> bool:
        # Does the model generate multimodal outputs beyond texts? It affects how we post-process the output.
        return False

    def __init__(self, cfg: Optional[Dict] = None):
        cfg = cfg or {}
        self.model = cfg.get('model', '').strip()
        generate_cfg = copy.deepcopy(cfg.get('generate_cfg', {}))
        self.max_retries = generate_cfg.pop('max_retries', 0)
        self.generate_cfg = generate_cfg
        self.model_type = cfg.get('model_type', '')

    def chat(
        self,
        messages: List[Union[Message, Dict]],
        stream: bool = True,
        extra_generate_cfg: Optional[Dict] = None,
    ) -> Union[List[Message], List[Dict], Iterator[List[Message]], Iterator[List[Dict]]]:
        """LLM chat interface.

        Args:
            messages: Inputted messages.
            functions: Inputted functions for function calling. OpenAI format supported.
            stream: Whether to use streaming generation.
            delta_stream: Whether to stream the response incrementally.
              (1) When False (recommended): Stream the full response every iteration.
              (2) When True: Stream the chunked response, i.e, delta responses.
            extra_generate_cfg: Extra LLM generation hyper-parameters.

        Returns:
            the generated message list response by llm.
        """

        # Unify the input messages to type List[Message]:
        messages = copy.deepcopy(messages)
        _return_message_type = 'dict'
        new_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                new_messages.append(Message(**msg))
            else:
                new_messages.append(msg)
                _return_message_type = 'message'
        messages = new_messages

        if not messages:
            raise ValueError('Messages can not be empty.')


        generate_cfg = merge_generate_cfgs(base_generate_cfg=self.generate_cfg, new_generate_cfg=extra_generate_cfg)
        if 'seed' not in generate_cfg:
            generate_cfg['seed'] = random.randint(a=0, b=2**30)

        if DEFAULT_SYSTEM_MESSAGE and messages[0].role != SYSTEM:
            messages = [Message(role=SYSTEM, content=DEFAULT_SYSTEM_MESSAGE)] + messages

        def _call_model_service():
            return self._chat(
                messages,
                stream=stream,
                generate_cfg=generate_cfg,
            )
        if stream:
            output = _call_model_service()
        else:
            output = retry_model_service_iterator(_call_model_service, max_retries=self.max_retries)

        # if isinstance(output, list):
        #     assert not stream
        #     logger.debug(f'LLM Output: \n{pformat([_.model_dump() for _ in output], indent=2)}')
        #     return self._convert_messages_to_target_type(output, _return_message_type)
        # else:
        #     assert stream
        return self._convert_messages_iterator_to_target_type(output, _return_message_type)

    def _chat(
        self,
        messages: List[Union[Message, Dict]],
        stream: bool,
        generate_cfg: dict,
    ) -> Union[List[Message], Iterator[List[Message]]]:
        if stream:
            return self._chat_stream(messages, generate_cfg=generate_cfg)
        else:
            return self._chat_no_stream(messages, generate_cfg=generate_cfg)

    @abstractmethod
    def _chat_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        raise NotImplementedError

    @abstractmethod
    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        raise NotImplementedError


    def _convert_messages_to_target_type(self, messages: List[Message],
                                         target_type: str) -> Union[List[Message], List[Dict]]:
        if target_type == 'message':
            return [Message(**x) if isinstance(x, dict) else x for x in messages]
        elif target_type == 'dict':
            return [x.model_dump() if not isinstance(x, dict) else x for x in messages]
        else:
            raise NotImplementedError

    def _convert_messages_iterator_to_target_type(
            self, messages_iter: Iterator[List[Message]],
            target_type: str) -> Union[Iterator[List[Message]], Iterator[List[Dict]]]:
        for messages in messages_iter:
            yield self._convert_messages_to_target_type(messages, target_type)


def retry_model_service(
    fn,
    max_retries: int = 10,
) -> Any:
    """Retry a function"""

    num_retries, delay = 0, 1.0
    while True:
        try:
            return fn()

        except ModelServiceError as e:
            num_retries, delay = _raise_or_delay(e, num_retries, delay, max_retries)


def retry_model_service_iterator(
    it_fn,
    max_retries: int = 10,
) -> Iterator:
    """Retry an iterator"""

    num_retries, delay = 0, 1.0
    while True:
        try:
            for rsp in it_fn():
                yield rsp
            break

        except ModelServiceError as e:
            num_retries, delay = _raise_or_delay(e, num_retries, delay, max_retries)


def _raise_or_delay(
    e: ModelServiceError,
    num_retries: int,
    delay: float,
    max_retries: int = 10,
    max_delay: float = 300.0,
    exponential_base: float = 2.0,
) -> Tuple[int, float]:
    """Retry with exponential backoff"""

    if max_retries <= 0:  # no retry
        raise e

    # Bad request, e.g., incorrect config or input
    if e.code == '400':
        raise e

    # If harmful input or output detected, let it fail
    if e.code == 'DataInspectionFailed':
        raise e
    if 'inappropriate content' in str(e):
        raise e

    # Retry is meaningless if the input is too long
    if 'maximum context length' in str(e):
        raise e

    logger.warning('ModelServiceError - ' + str(e).strip('\n'))

    if num_retries >= max_retries:
        raise ModelServiceError(exception=Exception(f'Maximum number of retries ({max_retries}) exceeded.'))

    num_retries += 1
    jitter = 1.0 + random.random()
    delay = min(delay * exponential_base, max_delay) * jitter
    time.sleep(delay)
    return num_retries, delay


def _rm_think(text: str) -> str:
    if '</think>' in text:
        return text.split('</think>')[-1].lstrip('\n')
    return text
