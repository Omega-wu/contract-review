import os
import traceback
from pathlib import Path

from loguru import logger

from model_config import CONFMAP
from pdf_parser.text_spliter.character_text_spliter import ModifyCharacterTextSplitter
from pdf_parser.utils.config_loader import load_config, initialize_tasks_and_models
from pdf_parser.utils.registry import TASK_REGISTRY

# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

TASK_NAME = "unstructed"
ocr_config_path = Path(__file__).parent.parent / "pdf_parser/configs/pdf2txt.yaml"
ocr_config = load_config(ocr_config_path)

# Get and assign values for the environment variables of the model path here.
ocr_config["ocr_model_weights"] = CONFMAP["OcrWeights"]
ocr_config["device"] = CONFMAP.get("DeviceMode", "cpu")
LAYOUT_URL = CONFMAP.get("layout_url")
if CONFMAP.get("DeviceMode", "cpu").startswith("npu") and "formula_recognition" in ocr_config["tasks"]:
    ocr_config["tasks"].pop("formula_recognition")
    ocr_config["tasks"].pop("region_detection")
    ocr_config["tasks"].pop("table_recognition")
if LAYOUT_URL:
    ocr_config["tasks"].pop("layout_detection")
tasks_dict = None


def init():
    from pdf_parser import tasks  # fmt: skip # noqa
    global tasks_dict

    logger.info("Start loading the OCR model.")

    tasks_dict = initialize_tasks_and_models(ocr_config)

    logger.info("The OCR model has been successfully loaded.")


def task(tasks_dict, input_data, file_info):
    layout_model = (
        tasks_dict["layout_detection"] if "layout_detection" in tasks_dict else None
    )
    region_model = (
        tasks_dict["region_detection"] if "region_detection" in tasks_dict else None
    )

    formula_rec_model = (
        tasks_dict["formula_recognition"]
        if "formula_recognition" in tasks_dict
        else None
    )
    ocr_model = (
        tasks_dict["ocr"]
        if "ocr" in tasks_dict
        else None
    )
    table_recognition_model = (
        tasks_dict["table_recognition"]
        if "table_recognition" in tasks_dict
        else None
    )
    # ocr_model = tasks_dict["ocr"].model if "ocr" in tasks_dict else None
    pdf_2_md_task = TASK_REGISTRY.get(TASK_NAME)(
        layout_model, region_model, formula_rec_model, ocr_model, table_recognition_model,
        ocr_config.get("batch_size", 1)
    )
    extract_results = pdf_2_md_task.process(input_data, file_info)

    return [extract_results]


def split_docs_document(res_list, file_info):
    chunk_size = file_info.get("split_length", 1)
    text_spliter = ModifyCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    documents = text_spliter.split_texts(res_list, file_info)
    return documents


def ocr_task(file_path, file_info, tasks_dict):
    file_directory = os.path.dirname(file_path)
    file_info.update({"file_directory": file_directory})
    file_info.update({"layout_url": LAYOUT_URL})
    res_list = task(tasks_dict, file_path, file_info)
    # file_name = file_info.get("file_name")
    # file_name_suffix = file_name.rsplit('.', 1)[1]
    # if file_name_suffix.lower() in ["ppt", "pptx"]:
    #     return res_list[0]
    documents = split_docs_document(res_list, file_info)

    return documents


def ocr_main(file_path, file_info=None):
    logger.info(f"Start executing the task.")
    try:
        res = ocr_task(file_path, file_info, tasks_dict)
        logger.info(f"Task done.")
        return res
    except Exception as e:
        logger.exception(f"Task execution failed: {e}")
        return [{"message": f"Task execution failed: {e}"}]


def ocr_test(file_path):
    init()
    res = ocr_main(file_path, file_info={})
    return res
