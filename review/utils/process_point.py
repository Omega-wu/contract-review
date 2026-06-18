import json
import re
from typing import List, Dict, Any, Union  # 引入Union支持多类型注解


def check_intersection_simple(a, b):
    """使用集合交集判断"""
    set_a = set(a)
    set_b = set(b)
    intersection = set_a & set_b  # 交集

    # 如果交集大小达到原字符串的80%，认为匹配
    similarity = len(intersection) / len(set_a)
    return similarity > 0.8


def _extract_table_name(audit_point: dict, ocr_result: dict) -> dict:
    """
    提取表格名称（用于review_exist任务）
    识别ocr结果是否有表名
    判断是否表名唯一 唯一的话直接查找 不是唯一的话还需要看

    Args:
        audit_point: 审核点数据
        ocr_result: OCR识别结果

    Returns:
        Dict: 包含表格名称的字典
    """

    # 定义表格名称关键词模式
    table_keywords = ["作业票", "记录", "表" , "清单"]

    # 从审核点获取表格名称
    table_title = audit_point.get("table_name", "")

    # 在OCR结果中查找表格名称
    # 这里需要根据你的具体OCR结果结构和表格命名规则来实现
    target_table_index = -1
    for i, item in enumerate(ocr_result):
        # 这里根据实际来定页面、坐标信息等
        page_content = item.get("page_content", "")
        chunk_type = item.get("metadata", {}).get("type", "").strip()
        location_info = item.get("metadata", {}).get("location", [{}])[0]
        page_num = location_info.get("page")
        # 检查文本块是否包含表格特征关键词
        if chunk_type in ["paragraph_title", "figure_title", "doc_title", "table_title"] and any(
                keyword in page_content for keyword in table_keywords):
            # 精炼表格名称（去除可能的多余字符）
            refined_name = _refine_table_name(page_content)
            # 检查是否匹配目标表格名称
            if check_intersection_simple(table_title, refined_name) or check_intersection_simple(table_title,
                                                                                                 page_content):
                target_table_index = i
                break
    if target_table_index == -1:
        return {
            "extraction_type": "table_name_with_keywords",
            "审核结果": "不合格",
            "审核依据": f"{table_title}缺失"
        }

    table_content_blocks = []
    next_index = target_table_index + 1
    while next_index < len(ocr_result):
        next_item = ocr_result[next_index]
        next_content = next_item.get("page_content", "").strip()
        next_type = next_item.get("metadata", {}).get("type", "").strip()
        # 检查是否是下一个表格名称（包含表格关键词）
        if next_type in ["paragraph_title", "figure_title", "doc_title", "table_title"] and any(
                keyword in next_content for keyword in table_keywords):
            break

        table_content_blocks.append(next_item)
        next_index += 1

    # 查找HTML表格内容
    html_table_blocks = []
    for content_block in table_content_blocks:
        page_content = content_block.get("page_content", "")
        label = content_block.get("metadata", {}).get("type", "").strip()
        if label == "abstract":
            html_table_blocks.append(content_block)
    if not html_table_blocks:
        return {
            "extraction_type": "table_name_with_keywords",
            "审核结果": "不合格",
            "审核依据": f"{table_title}缺失"
        }

    return {
        "extraction_type": "table_name_with_keywords",
        "审核结果": "合格",
        "审核依据": f"{table_title}存在，没有缺失",
        "page_num": page_num
    }


def _extract_table_and_fields(audit_point: dict, ocr_result: dict) -> dict:
    """
    根据表格标题和表格内字段进行联合定位匹配（用于review_table任务）
    在OCR结果中查找同时包含指定表格标题和字段关键词的表格

    Args:
        audit_point: 审核点数据，包含表格名称和字段关键词
        ocr_result: OCR识别结果，包含HTML表格内容和位置信息

    Returns:
        Dict: 包含表格名称和字段内容的字典
    """

    # 定义表格名称关键词模式
    table_keywords = ["作业票", "记录", "表", "清单"]

    # 从审核点获取表格名称和字段关键词
    table_title = audit_point.get("table_name", "").strip()
    field_keyword = audit_point.get("table_field", "")
    if field_keyword:
        field_keyword = field_keyword.strip()
    # 查找目标表格名称的位置
    target_table_index = -1

    for i, item in enumerate(ocr_result):
        page_content = item.get("page_content", "").strip()
        chunk_type = item.get("metadata", {}).get("type", "").strip()
        # 检查文本块是否包含表格特征关键词
        if chunk_type in ["paragraph_title", "figure_title", "doc_title", "table_title"] and any(
                keyword in page_content for keyword in table_keywords):
            # 精炼表格名称（去除可能的多余字符）
            refined_name = _refine_table_name(page_content)
            # 检查是否匹配目标表格名称
            if check_intersection_simple(table_title, refined_name) or check_intersection_simple(table_title,
                                                                                                 page_content):
                target_table_index = i
                break
    if target_table_index == -1:
        return {
            "extraction_type": "table_fields",
            "html_results": [],
            "page_nums": []
        }

    # 提取目标表格的内容（直到下一个表格名称或结束）
    table_content_blocks = []
    next_index = target_table_index + 1

    while next_index < len(ocr_result):
        next_item = ocr_result[next_index]
        next_content = next_item.get("page_content", "").strip()
        next_type = next_item.get("metadata", {}).get("type", "").strip()
        # 检查是否是下一个表格名称（包含表格关键词）
        if next_type in ["paragraph_title", "figure_title", "doc_title", "table_title"] and any(
                keyword in next_content for keyword in table_keywords):
            break

        table_content_blocks.append(next_item)
        next_index += 1

    # 查找HTML表格内容
    html_table_blocks = []
    for content_block in table_content_blocks:
        page_content = content_block.get("page_content", "")
        label = content_block.get("metadata", {}).get("type", "").strip()
        if label == "abstract":
            html_table_blocks.append(content_block)
            print("abstract======", content_block)

        # 提取字段信息
    page_nums = []
    html_results = []
    if field_keyword:  # 表锁定，关键字得唯一
        for html_block in html_table_blocks:
            html_content = html_block.get("page_content", "")
            location_info = html_block.get("metadata", {}).get("location", [{}])[0]
            page_num = location_info.get("page", 0)
            if field_keyword in html_content:
                html_results.append(html_block)
                page_nums.append(page_num)

        # if not page_nums and not html_results:
        #     for html_block in html_table_blocks:
        #         html_content = html_block.get("page_content", "")
        #         location_info = html_block.get("metadata", {}).get("location", [{}])[0]
        #         page_num = location_info.get("page", 0) + 1
        #         html_results.append(html_content)
        #         page_nums.append(page_num)

    return {
        "extraction_type": "table_fields",
        "html_results": html_results,
        "page_nums": page_nums
    }


def _refine_table_name(text: str) -> str:
    """
    精炼表格名称，去除可能的多余空格

    Args:
        text: 原始文本

    Returns:
        str: 精炼后的表格名称
    """
    import re

    # 去除多余的空格和标点
    text = re.sub(r'\s+', '', text).strip()

    return text


def process_audit_point(audit_point: dict, ocr_result: List[dict]) -> dict:
    """
    根据审核点的task类型处理审核点

    Args:
        audit_point: 审核点数据，应包含task、审核关键字等字段
        ocr_result: OCR识别结果

    Returns:
        Dict: 处理后的审核点信息，包含提取的内容
    """
    # 获取task类型，默认为review_table
    task_type = audit_point.get("task", "review_table")

    # 基础结果结构
    result = {
        "audit_point": {},  # 包含所有审核点信息
        "processed_content": {}  # ocr内容
    }

    # 根据task类型进行不同处理
    if task_type == "review_exist":
        # 只需要提取表格名称
        result["processed_content"] = _extract_table_name(audit_point, ocr_result)
        result["audit_point"] = audit_point

    elif task_type == "review_table":
        result["audit_point"] = audit_point
        result["processed_content"] = _extract_table_and_fields(audit_point, ocr_result)

    return result
