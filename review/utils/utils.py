import base64
import copy
import hashlib
import os
import re
import sys
import time
import traceback
import urllib.parse
from io import BytesIO
from typing import Any, List, Literal, Optional, Tuple, Union

import json
import requests
from pydantic import BaseModel

from review.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE,  SYSTEM, USER, ContentItem, Message
from review.utils.log import logger



def hash_sha256(text: str) -> str:
    hash_object = hashlib.sha256(text.encode())
    key = hash_object.hexdigest()
    return key


def print_traceback(is_error: bool = True):
    tb = ''.join(traceback.format_exception(*sys.exc_info(), limit=3))
    if is_error:
        logger.error(tb)
    else:
        logger.warning(tb)



def get_basename_from_url(path_or_url: str) -> str:
    if re.match(r'^[A-Za-z]:\\', path_or_url):
        # "C:\\a\\b\\c" -> "C:/a/b/c"
        path_or_url = path_or_url.replace('\\', '/')

    parsed = urllib.parse.urlparse(path_or_url)

    # 优先检查查询参数中的文件路径（如fileUrl参数）
    query_params = urllib.parse.parse_qs(parsed.query)
    for key in ['fileUrl', 'file', 'filename', 'path']:  # 常见文件参数名
        if key in query_params:
            # 取最后一个值（因为parse_qs返回列表）
            file_path = query_params[key][-1]
            basename = os.path.basename(file_path)
            if basename:
                return urllib.parse.unquote(basename).strip()
    # "/mnt/a/b/c" -> "c"
    # "https://github.com/here?k=v" -> "here"
    # "https://github.com/" -> ""
    basename = parsed.path
    basename = os.path.basename(basename)
    basename = urllib.parse.unquote(basename)
    basename = basename.strip()

    # "https://github.com/" -> "" -> "github.com"
    if not basename:
        basename = [x.strip() for x in path_or_url.split('/') if x.strip()][-1]

    return basename


def is_http_url(path_or_url: str) -> bool:
    if path_or_url.startswith('https://') or path_or_url.startswith('http://'):
        return True
    return False


def is_image(path_or_url: str) -> bool:
    filename = get_basename_from_url(path_or_url).lower()
    for ext in ['jpg', 'jpeg', 'png', 'webp']:
        if filename.endswith(ext):
            return True
    return False




def save_url_to_local_work_dir(url: str, save_dir: str, save_filename: str = '') -> str:
    if not save_filename:
        save_filename = get_basename_from_url(url)
    new_path = os.path.join(save_dir, save_filename)
    if os.path.exists(new_path):
        os.remove(new_path)
    logger.info(f'Downloading {url} to {new_path}...')
    start_time = time.time()
    headers = {
        'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        with open(new_path, 'wb') as file:
            file.write(response.content)
    else:
        raise ValueError('Can not download this file. Please check your network or the file link.')
    end_time = time.time()
    logger.info(f'Finished downloading {url} to {new_path}. Time spent: {end_time - start_time} seconds.')
    return new_path


def save_text_to_file(path: str, text: str) -> None:
    with open(path, 'w', encoding='utf-8') as fp:
        fp.write(text)


def contains_html_tags(text: str) -> bool:
    pattern = r'<(p|span|div|li|html|script)[^>]*?'
    return bool(re.search(pattern, text))


def get_content_type_by_head_request(path: str) -> str:
    try:
        response = requests.head(path, timeout=5)
        content_type = response.headers.get('Content-Type', '')
        return content_type
    except requests.RequestException:
        return 'unk'


def get_file_type(path: str) -> Literal['pdf', 'doc', 'docx', 'pptx', 'txt', 'html', 'csv', 'tsv', 'xlsx', 'xls', 'unk']:
    f_type = get_basename_from_url(path).split('.')[-1].lower()
    if f_type in ['pdf', 'doc', 'docx', 'pptx', 'csv', 'tsv', 'xlsx', 'xls', 'png', 'ofd', 'jpg']:
        # Specially supported file types
        return f_type


class PydanticJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        return super().default(obj)


def json_dumps_pretty(obj: dict, ensure_ascii=False, indent=2, **kwargs) -> str:
    return json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent, cls=PydanticJSONEncoder, **kwargs)


def json_dumps_compact(obj: dict, ensure_ascii=False, indent=None, **kwargs) -> str:
    return json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent, cls=PydanticJSONEncoder, **kwargs)



def extract_files_from_messages(messages: List[Message], include_images: bool) -> List[str]:
    files = []
    for msg in messages:
        if isinstance(msg.content, list):
            for item in msg.content:
                if item.file and item.file not in files:
                    files.append(item.file)
                if include_images and item.image and item.image not in files:
                    files.append(item.image)
    return files


def merge_generate_cfgs(base_generate_cfg: Optional[dict], new_generate_cfg: Optional[dict]) -> dict:
    generate_cfg: dict = copy.deepcopy(base_generate_cfg or {})
    if new_generate_cfg:
        for k, v in new_generate_cfg.items():
            if k == 'stop':
                stop = generate_cfg.get('stop', [])
                stop = stop + [s for s in v if s not in stop]
                generate_cfg['stop'] = stop
            else:
                generate_cfg[k] = v
    return generate_cfg



def encode_image_as_base64(path: str, max_short_side_length: int = -1) -> str:
    from PIL import Image
    image = Image.open(path)

    if (max_short_side_length > 0) and (min(image.size) > max_short_side_length):
        ori_size = image.size
        image = resize_image(image, short_side_length=max_short_side_length)
        logger.debug(f'Image "{path}" resized from {ori_size} to {image.size}.')

    image = image.convert(mode='RGB')
    buffered = BytesIO()
    image.save(buffered, format='JPEG')
    return 'data:image/jpeg;base64,' + base64.b64encode(buffered.getvalue()).decode('utf-8')



def encode_video_as_base64(path: str) -> str:
    with open(path, 'rb') as video_file:
        return 'data:;base64,' + base64.b64encode(video_file.read()).decode('utf-8')


def load_image_from_base64(image_base64: Union[bytes, str]):
    from PIL import Image
    image = Image.open(BytesIO(base64.b64decode(image_base64)))
    image.load()
    return image


def resize_image(img, short_side_length: int = 1080):
    from PIL import Image
    assert isinstance(img, Image.Image)

    width, height = img.size

    if width <= height:
        new_width = short_side_length
        new_height = int((short_side_length / width) * height)
    else:
        new_height = short_side_length
        new_width = int((short_side_length / height) * width)

    resized_img = img.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
    return resized_img


def get_last_usr_msg_idx(messages: List[Union[dict, Message]]) -> int:
    i = len(messages) - 1
    while (i >= 0) and (messages[i]['role'] != 'user'):
        i -= 1
    assert i >= 0, messages
    assert messages[i]['role'] == 'user'
    return i


def rm_default_system(messages: List[Message]) -> List[Message]:
    if len(messages) > 1 and messages[0].role == SYSTEM:
        if isinstance(messages[0].content, str):
            if messages[0].content.strip() == DEFAULT_SYSTEM_MESSAGE:
                return messages[1:]
            else:
                return messages
        elif isinstance(messages[0].content, list):
            if len(messages[0].content) == 1 and messages[0].content[0].text.strip() == DEFAULT_SYSTEM_MESSAGE:
                return messages[1:]
            else:
                return messages
        else:
            raise TypeError
    else:
        return messages
