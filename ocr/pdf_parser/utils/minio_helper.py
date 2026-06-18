import random
import string

import requests
from loguru import logger

from model_config import CONFMAP


def get_pic_random_name():
    """
    随机生成一张图片名,生成一个10字符的随机字符串,包含小写字母和数字
    可能生成 'nO4'
    """
    chars = (
        string.ascii_letters
        + string.digits
        + "这里预留一定量中文数据目的是为了降低生成字符串重复概率"
    )
    random_chars = [random.choice(chars) for i in range(3)]

    random_filename = "".join(random_chars)
    filename = f"{random_filename}"

    return filename
