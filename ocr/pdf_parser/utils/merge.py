"""
合并成txt文件
"""

import os
import re
import json
import base64
from io import BytesIO
from traceback import format_exc
from PIL import Image

from loguru import logger


def image_to_base64(image: Image.Image, fmt='PNG') -> list:
    try:
        output_buffer = BytesIO()
        print(image)
        image.save(output_buffer, format=fmt)
        byte_data = output_buffer.getvalue()
        base64_data = base64.b64encode(byte_data).decode('utf-8')
        # print(base64_data)
        return [{'base64': base64_data, 'format': '.png'}]
    except Exception as e:
        print(e)


def get_file_title(path):
    """获取 文件名（纯净版）

    Args:
        path (str): 原始文件路径

    Returns:
        str: 文件名（纯净版）
    """
    base_name = os.path.basename(path)
    # 获取文件名
    file_base_name, suffix = os.path.splitext(base_name)
    # 去掉文件后缀格式
    base_name2 = base_name.replace(suffix, "")
    # 去掉 "数字."
    last_name = re.sub(r"^\d+\.", "", base_name2)

    return last_name


def process_rule(text, last_name):
    try:
        is_match = False

        if not isinstance(text, str):
            return is_match, text

        """针对规章制度条文，进行适配

        Args:
            text (str): 原始文本
            last_name (str): 纯净版的文件名

        Returns:
            _type_: 适配后文本
        """
        # 是否是"第x章"或"第x条"的格式
        chapter_or_article_pattern = re.compile(r"(\n?第[一二三四五六七八九十\d]+[章条])")

        # 只对第一个匹配进行处理
        match = re.search(chapter_or_article_pattern, text)
        if match:
            is_match = True
            isMoreLine = False
            # 拼接 文件名和换行符，并在第一个匹配项后加换行符
            match_str = match.group(0)
            if isMoreLine:
                text = f"\n{match_str}\n" + "    " + text[match.end():]
            else:
                text = f"{match_str}\n" + "    " + text[match.end():]
    except:
        # logger.info(f"process rule:{format_exc()}")
        pass

    return is_match, text


def merge_partitions(partitions, fpath, file_info):
    try:
        """针对ocr识别的对象列表进行合并成txt--主要的RAG策略

        Args:
            partitions ([], optional): ocr识别的对象结果列表
            fpath (str, optional): 文件路径

        Returns:
            _type_: 文本内容，元数据信息
        """
        doc_content = []
        is_first_elem = True
        last_label = ""
        start_indexes = 0
        # last_name = get_file_title(fpath)

        metadata = dict(
            bboxes=[],
            pages=[],
            indexes=[],
            types=[],
            sources=[],
            page_height=[],
            page_width=[],
        )

        # Add default file name if no partitions exist
        if not partitions:
            file_name = os.path.basename(fpath) if fpath else "unknown"
            metadata["sources"].append(file_name)
            return "", metadata

        for page in partitions:
            file_name = os.path.basename(page["input_path"])
            page_num = page["page_index"]
            page_height = page["doc_preprocessor_res"]["output_img"].shape[0]
            page_width = page["doc_preprocessor_res"]["output_img"].shape[1]
            for part in page["parsing_res_list"]:
                if part.label in ["paragraph_title", "figure_title", "doc_title", "text", "table", "content",
                                  "abstract","table_title"]:
                    bbox, text, label, image = part.bbox, part.content.strip().replace("\n", " "), part.label, part.image
                    # 如果是规章制度
                    # is_match, text = process_rule(text, last_name=last_name)

                    if is_first_elem:
                        if label in ["image", "chart"]:
                            res_list = image_to_base64(image["img"])
                            file_id = file_info.get("file_id", None)
                            image_data = {"picture_obj_list": res_list, "file_id": file_id}
                            doc_content.append(f"image[{json.dumps(image_data)}]")
                        else:
                            f_text = (
                                f"{text}\n"
                                if label in ["paragraph_title", "doc_title"]
                                else text
                            )
                            doc_content.append(f_text)
                        is_first_elem = False
                    else:
                        if last_label in ["paragraph_title", "doc_title"] and label in [
                            "paragraph_title",
                            "doc_title",
                        ]:
                            text = f"\n{text}"
                        elif label in ["paragraph_title", "doc_title"]:
                            # 标题 添加 文档名称
                            text = f"\n\n{text}"
                        elif label == "table":
                            text = f"\n\n{text}"
                        elif label in ["image", "chart"]:
                            res_list = image_to_base64(image["img"])
                            file_id = file_info.get("file_id", None)
                            image_data = {"picture_obj_list": res_list, "file_id": file_id}
                            text = f"image[{json.dumps(image_data)}]"
                        else:
                            if last_label == "table":
                                text = f"\n\n{text}"
                            else:
                                text = f"\n{text}"
                        doc_content.append(text)

                    last_label = label
                    metadata["bboxes"].append(bbox)
                    metadata["pages"].append(page_num)
                    metadata["types"].append(label)
                    metadata["sources"].append(file_name)
                    indexes = [start_indexes, start_indexes + len(doc_content[-1]) - 1]
                    metadata["indexes"].append(indexes)
                    metadata["page_height"].append(page_height)
                    metadata["page_width"].append(page_width)
                    start_indexes += len(doc_content[-1])

        content = "".join(doc_content)
        print(content)
        logger.info(f"{content}")
    except:
        logger.error(f"merge_partitions:{format_exc()}")
        content = ""
        # 返回正确的默认结构，而不是空字典
        last_name = get_file_title(fpath) if fpath else "unknown_file"
        metadata = dict(
            bboxes=[],
            pages=[],
            indexes=[],
            types=[],
            sources=[last_name],  # 至少包含一个文件名
            page_height=[],
            page_width=[],
        )

    return content, metadata


def merge_partitions_ppt(partitions, fpath, file_info):
    try:
        documents = []
        for sort_index, page in enumerate(partitions):
            doc_content = []
            is_first_elem = True
            last_label = ""
            file_name = os.path.basename(page["input_path"])
            page_num = page["page_index"]
            page_height = page["doc_preprocessor_res"]["output_img"].shape[0]
            page_width = page["doc_preprocessor_res"]["output_img"].shape[1]
            metadata = {"parent_title": file_name,
                        "multi_title": sort_index,
                        "title": "",
                        "sort": sort_index,
                        "file_name": file_name,
                        "md5": file_info.get("md5"),
                        "file_id": file_info.get("file_id"),
                        "partition_key": file_info.get("partition_key"),
                        "file_directory": file_info.get("file_directory"),
                        "content": "",
                        "content_type": "ocr",
                        "location": []}

            for part in page["parsing_res_list"]:
                if part.label in ["paragraph_title", "doc_title", "text", "image", "formula", "table",
                                  "chart", "content"]:
                    bbox, text, label, image = part.bbox, part.content.strip(), part.label, part.image

                    if is_first_elem:
                        if label in ["image", "chart"]:
                            res_list = image_to_base64(image["img"])
                            file_id = file_info.get("file_id", None)
                            image_data = {"picture_obj_list": res_list, "file_id": file_id}
                            doc_content.append(f"image[{json.dumps(image_data)}]")
                        else:
                            f_text = (
                                f"{text} \n"
                                if label in ["paragraph_title", "doc_title"]
                                else text
                            )
                            doc_content.append(f_text)
                        is_first_elem = False
                    else:
                        if last_label in ["paragraph_title", "doc_title"] and label in [
                            "paragraph_title",
                            "doc_title",
                        ]:
                            text = f"\n{text}"
                        elif label in ["paragraph_title", "doc_title"]:
                            # 标题 添加 文档名称
                            text = f"\n\n{text}"
                        elif label == "table":
                            text = f"\n\n{text}"
                        elif label in ["image", "chart"]:
                            res_list = image_to_base64(image["img"])
                            file_id = file_info.get("file_id", None)
                            image_data = {"picture_obj_list": res_list, "file_id": file_id}
                            text = f"image[{json.dumps(image_data)}]"
                        else:
                            if last_label == "table":
                                text = f"\n\n{text}"
                            else:
                                text = f"\n{text}"
                        doc_content.append(text)

                    last_label = label

                    leftTop = {"x": bbox[0], "y": bbox[1]}
                    rightBottom = {"x": bbox[2], "y": bbox[3]}
                    metadata["location"].append(
                        {'page': page_num,
                         'width': page_width,
                         'height': page_height,
                         'leftTop': leftTop,
                         'rightBottom': rightBottom
                         })

            content = "".join(doc_content)
            metadata["content"] = content
            print("*************************")
            print(content)
            print(metadata)
            print("*************************")
            new_doc = dict(page_content=content, metadata=metadata)
            documents.append(new_doc)

    except:
        logger.error(f"merge_partitions:{format_exc()}")
        documents = []
        file_name = os.path.basename(page["input_path"]) if fpath else "unknown_file"
        content = ""
        metadata = {"parent_title": file_name,
                    "multi_title": 0,
                    "title": "",
                    "sort": 0,
                    "file_name": file_name,
                    "md5": file_info.get("md5"),
                    "file_id": file_info.get("file_id"),
                    "partition_key": file_info.get("partition_key"),
                    "file_directory": file_info.get("file_directory"),
                    "content": content,
                    "content_type": "ocr",
                    "location": []}
        new_doc = dict(page_content=content, metadata=metadata)
        documents.append(new_doc)

    return documents
