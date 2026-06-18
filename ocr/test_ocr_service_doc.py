import os
import requests
import json
import subprocess
from typing import Literal
import re
import urllib.parse

# 测试OCR服务的PDF处理功能
def test_ocr_pdf(pdf_path):
    url = "http://127.0.0.1:8001/ocr/task"
    file_info = {
        "file_id": "test_file_001"
    }
    files = {
        'file': open(pdf_path, 'rb'),
        'file_info': (None, json.dumps(file_info))
    }
    print(files)
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        result = response.json()
        print(result)
        
        # 保存结果到JSON文件
        output_path = pdf_path.replace('.pdf', '_result.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        print(f"OCR处理完成，结果已保存到: {output_path}")
        return True
    except Exception as e:
        print(f"OCR处理失败: {str(e)}")
        return False

def convert_doc_to_pdf(doc_path, output_dir=None):
    """
    使用LibreOffice命令行将Word文档转换为PDF
    :param doc_path: 输入的Word文档路径(.doc或.docx)
    :param output_dir: 输出的PDF目录，默认为Word文档所在目录
    :return: 转换成功返回True，否则返回False
    """
    doc_extension = os.path.splitext(doc_path)[1]
    print(doc_extension)
    if doc_extension == ".pdf":
        return True
    if doc_extension in [".doc", ".docx"]:
        # 检查输入文件是否存在
        if not os.path.exists(doc_path):
            print(f"错误：文件不存在 - {doc_path}")
            return False

        # 如果未指定输出目录，则使用输入文件所在目录
        if output_dir is None:
            output_dir = os.path.dirname(doc_path)
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        try:
            # 构建命令
            # 注意：在某些系统上，命令可能是 'libreoffice7.0' 等带版本号的形式[8](@ref)
            cmd = [
                'libreoffice',  # 或 'libreoffice7.0', 'soffice'
                '--headless',  # 无界面模式
                '--convert-to',
                'pdf',
                '--outdir',
                output_dir,
                doc_path
            ]

            # 执行转换命令
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
            print(f"转换成功！PDF文件保存在：{output_dir}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"转换失败，LibreOffice返回错误：{e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            print("转换超时")
            return False
        except FileNotFoundError:
            print("未找到LibreOffice，请确保已安装并将其添加到系统PATH环境变量中")
            return False
        except Exception as e:
            print(f"转换过程中发生未知错误：{e}")
            return False

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

def get_file_type(path: str) -> Literal['pdf', 'doc', 'docx', 'pptx', 'txt', 'html', 'csv', 'tsv', 'xlsx', 'xls', 'unk']:
    f_type = get_basename_from_url(path).split('.')[-1].lower()
    if f_type in ['pdf', 'doc', 'docx', 'pptx', 'csv', 'tsv', 'xlsx', 'xls', 'png', 'ofd', 'jpg']:
        # Specially supported file types
        return f_type

if __name__ == "__main__":
    file_path = r"/data/doc_review/test_pdf/test/动火作业办理说明.docx"
    file_type = get_file_type(file_path)
    print("file_type==========", file_type)
    if file_type in ["doc", "docx"]:
        convert_doc_to_pdf(file_path)
        file_path = file_path.replace(f".{file_type}", ".pdf")
    print("file_path==========", file_path)
    test_ocr_pdf(file_path)
