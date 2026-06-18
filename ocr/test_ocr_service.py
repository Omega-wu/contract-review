import requests
import json

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


if __name__ == "__main__":
    pdf_file = r"/data/doc_review/test_pdf/动土作业票.pdf"
    test_ocr_pdf(pdf_file)
