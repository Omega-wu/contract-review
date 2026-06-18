import json
import uuid
from pathlib import Path
from pprint import pformat

from flask import Blueprint, request
from langchain_core.documents import Document
from loguru import logger
from pydantic import BaseModel, Field, field_serializer

from api.file_util import rm_file, is_valid_pdf
from tasks.pdf2txt import ocr_main

ocr = Blueprint("ocr", __name__, url_prefix="/ocr")


# 定义 Pydantic 模型
class FileInfo(BaseModel):
    file_id: str


class Task(BaseModel):
    task_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    file_path: str
    file_info: FileInfo

    @field_serializer("task_id", mode="plain")
    def serialize_id(self, task_id):
        return str(task_id)


@ocr.post("/task")
def ocr_task() -> list[Document]:
    # 获取JSON参数
    file_info_str = request.form.get("file_info")
    if file_info_str:
        file_info = json.loads(file_info_str)
    else:
        return [{"message": "未发送 file_info 字段"}]

    logger.info(f"request file_info: \n{pformat(file_info)}")

    uploaded_file = request.files.get("file")
    if uploaded_file and uploaded_file.filename:
        filename = uploaded_file.filename

        # 保存上传的文件
        upload_dir = Path(__file__).parent.parent / "download"
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / f"{uuid.uuid4().hex}_{filename}"
        uploaded_file.save(str(file_path))

        # 验证PDF文件
        # if not is_valid_pdf(str(file_path)):
        #     rm_file(str(file_path))
        #     return [{"message": "不是有效的PDF文件"}]

        try:
            res = ocr_main(str(file_path), file_info)
            return res
        finally:
            rm_file(str(file_path))

    # 没有文件上传，返回错误
    else:
        return [{"message": "请上传PDF文件"}]
