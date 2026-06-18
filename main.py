from review.tasks.review import review_chat
import os
from pathlib import Path


def process_directory(directory_path, allowed_extensions=None):
    """
    处理目录中的所有文件，组装成 messages 格式

    Args:
        directory_path (str): 目录路径
        allowed_extensions (list, optional): 允许的文件扩展名，如 ['.pdf', '.docx']

    Returns:
        list: 组装好的 messages 列表
    """
    # 默认允许所有文件类型
    if allowed_extensions is None:
        allowed_extensions = []

    # 将路径转换为 Path 对象
    dir_path = Path(directory_path)

    # 检查目录是否存在
    if not dir_path.exists():
        print(f"错误：目录 {directory_path} 不存在")
        return []

    if not dir_path.is_dir():
        print(f"错误：{directory_path} 不是一个目录")
        return []

    # 获取目录中的所有文件
    files = []
    for file_path in dir_path.iterdir():
        if file_path.is_file():
            # 如果指定了文件类型过滤
            if allowed_extensions:
                if file_path.suffix.lower() in [ext.lower() for ext in allowed_extensions]:
                    files.append(file_path)
            else:
                files.append(file_path)

    if not files:
        print(f"目录 {directory_path} 中没有符合条件的文件")
        return []

    # 按文件名排序（可选）
    files.sort()

    # 组装 messages
    file_contents = [{"file": str(file_path)} for file_path in files]
    messages = [
        {"role": "user", "content": file_contents}
    ]

    # 打印处理信息
    print(f"找到 {len(files)} 个文件:")
    for i, file_path in enumerate(files, 1):
        print(f"  {i}. {file_path.name}")

    return messages


# 使用示例
if __name__ == "__main__":
    directory_path = r"/data/doc_review/test_pdf"

    # 方法1：处理所有文件
    messages = process_directory(directory_path)

    # 方法2：只处理特定类型的文件
    # messages = process_directory(directory_path, allowed_extensions=['.pdf', '.docx'])

    if messages:
        # 调用处理函数
        res = list(review_chat(messages))
        print(res)