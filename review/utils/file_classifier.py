import re
import logging
from typing import Dict, List


class FileClassifier:
    """
    文件分类器：根据正则表达式从文件名提取类型，并与审核点表格中的文档类型匹配。
    """

    # 定义文件名正则模式与分类的映射
    # 模式键：一个简短标识
    # 模式值：一个元组 (正则表达式模式, 返回的分类标签)
    FILE_PATTERNS = {
        'earthwork': (r'动土', '动土安全类'),
        'fire_level1': (r'一级.*动火', '一级动火作业类'),
        'fire_level2': (r'二级.*动火', '二级动火作业类'),
        'fire_level3': (r'特级.*动火', '特级动火作业类'),
        'confined_space': (r'受限空间', '受限空间类'),
        'hoisting_level1': (r'一级.*吊装', '一级吊装作业类'),
        'hoisting_level2': (r'二级.*吊装', '二级吊装作业类'),
        'hoisting_level3': (r'三级.*吊装', '三级吊装作业类'),
        'circuit_breaking': (r'断路', '断路作业类'),
        'pipeline': (r'管线', '管线打开(盲板抽堵)类'),
        'ray': (r'射线', '非特殊高风险类作业(射线作业)'),
        'scaffolding': (r'脚手架搭建', '非特殊高风险类作业（脚手架作业)'),
        'high_altitude_level1': (r'一级.*高处', '一二级高处作业类'),
        'high_altitude_level2': (r'二级.*高处', '一二级高处作业类'),
        'high_altitude_level3': (r'三级.*高处', '三四级高处作业类'),
        'high_altitude_level4': (r'四级.*高处', '三四级高处作业类'),
        'temporary_electricity': (r'临时用电', '临时用电作业类')
        # 可以继续添加更多模式...
    }

    def __init__(self, custom_patterns: Dict = None):
        """
        初始化文件分类器

        Args:
            custom_patterns: 自定义的正则模式字典，可覆盖或补充默认模式
        """
        self.patterns = self.FILE_PATTERNS.copy()
        if custom_patterns:
            self.patterns.update(custom_patterns)

        # 预编译所有正则表达式以提高性能
        self.compiled_patterns = {}
        for pattern_key, (pattern_str, category) in self.patterns.items():
            try:
                self.compiled_patterns[pattern_key] = (
                    re.compile(pattern_str),
                    category
                )
            except re.error as e:
                logging.error(f"正则表达式编译错误 '{pattern_str}': {e}")

    def classify_filename(self, filename: str) -> Dict[str, str]:
        """
        对单个文件名进行分类

        Args:
            filename: 要分类的文件名

        Returns:
            Dict: 包含分类结果的字典
        """
        # 移除可能的文件扩展名以便更好地匹配
        name_only = filename.rsplit('.', 1)[0] if '.' in filename else filename
        logging.info(f"分类文件名: {filename} -> {name_only}")

        for pattern_key, (compiled_pattern, category) in self.compiled_patterns.items():
            match = compiled_pattern.search(name_only)
            if match:
                logging.info(f"匹配到模式 '{pattern_key}': 分类为 '{category}'")
                return {
                    'original_filename': filename,
                    'matched_pattern': pattern_key,
                    'category': category,
                    'matched_text': match.group(),
                    'match_position': f"{match.start()}-{match.end()}"
                }

        # 如果没有匹配到任何模式
        logging.warning(f"文件名 '{filename}' 未匹配到任何分类模式")
        return {
            'original_filename': filename,
            'matched_pattern': None,
            'category': '未分类',
            'matched_text': None,
            'match_position': None
        }

    def batch_classify(self, filenames: List[str]) -> Dict[str, List]:
        """
        批量分类文件名

        Args:
            filenames: 文件名列表

        Returns:
            Dict: 按分类组织的结果
        """
        results = {
            'classified': {},
            'unclassified': []
        }

        for filename in filenames:
            classification = self.classify_filename(filename)
            category = classification['category']

            if category == '未分类':
                results['unclassified'].append(classification)
            else:
                if category not in results['classified']:
                    results['classified'][category] = []
                results['classified'][category].append(classification)

        return results
