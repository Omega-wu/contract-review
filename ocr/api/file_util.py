import os
import time
from pathlib import Path
from traceback import format_exc

import pypdfium2
import requests
from loguru import logger

target_dir = (Path(__file__).parent.parent / "download").absolute()


# 清理临时文件
def rm_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


# 文件合法性检查
def is_valid_pdf(path: str) -> bool:
    try:
        pdf = pypdfium2.PdfDocument(path)
        pdf.get_page(0)
    except Exception:
        return False
    else:
        return True
