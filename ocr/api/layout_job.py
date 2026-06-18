import json
import uuid
from pathlib import Path
from pprint import pformat
import pickle
import base64
from flask import Blueprint, request, jsonify
from loguru import logger
from pydantic import BaseModel, Field, field_serializer

from api.file_util import rm_file, is_valid_pdf
from tasks.layout import layout_main

layout = Blueprint("layout", __name__, url_prefix="/layout")


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


@layout.post("/task")
def layout_task():
    data = request.json

    if 'array_pickled' not in data:
        return jsonify({'error': 'No array data provided'}), 400

    try:
        # 解码并反序列化
        print("===============================")
        array_base64 = data['array_pickled']
        #print("=====================array_base64", array_base64)
        array_pickled = base64.b64decode(array_base64)
        #print(array_pickled)
        image_array = pickle.loads(array_pickled)
        #print(image_array)
        print(image_array[0].shape)
        print(type(image_array[0]))
        res = layout_main(image_array)
        return res

    except Exception as e:
        return jsonify({'error': str(e)}), 400
