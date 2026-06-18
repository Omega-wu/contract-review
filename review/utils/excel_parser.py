import pandas as pd
import json


def excel_to_json(excel_file, sheet_name=0, orient='records'):
    """
    将Excel文件转换为JSON格式

    参数:
    - excel_file: Excel文件路径
    - sheet_name: 工作表名称或索引，默认为第一个工作表
    - orient: JSON格式方向，可选'records', 'index', 'values', 'table'等

    返回:
    - 成功时返回JSON对象，失败时返回错误信息字典
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_file, sheet_name=sheet_name)

        # 处理日期时间列，将其转换为字符串格式
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

        # 转换为JSON字符串
        json_str = df.to_json(orient=orient, force_ascii=False, indent=2)

        # 转换为JSON对象
        return json.loads(json_str)

    except FileNotFoundError:
        return {"error": f"文件未找到: {excel_file}"}
    except Exception as e:
        return {"error": f"处理Excel文件时出错: {str(e)}"}
