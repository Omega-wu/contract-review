from pathlib import Path

from loguru import logger

from model_config import CONFMAP
from pdf_parser.utils.config_loader import load_config, initialize_tasks_and_models

TASK_NAME = "layout_detection"
ocr_config_path = Path(__file__).parent.parent / "pdf_parser/configs/layout.yaml"
ocr_config = load_config(ocr_config_path)

# Get and assign values for the environment variables of the model path here.
ocr_config["ocr_model_weights"] = CONFMAP["OcrWeights"]
#ocr_config["device"] = CONFMAP.get("DeviceMode", "cpu")
ocr_config["device"] = "npu"
tasks_dict = None


def init():
    from pdf_parser import tasks  # fmt: skip # noqa
    global tasks_dict

    logger.info("Start loading the layout model.")

    tasks_dict = initialize_tasks_and_models(ocr_config)

    logger.info("The layout model has been successfully loaded.")


def layout_task(input_data):
    layout_instance = tasks_dict[TASK_NAME] if TASK_NAME in tasks_dict else None
    res = layout_instance.predict(input_data)
    return res


def layout_main(input_data):
    logger.info(f"Start executing the task.")
    try:
        res = layout_task(input_data)
        logger.info(f"Task done.")
        return res
    except Exception as e:
        logger.exception(f"Task execution failed: {e}")
        return [{"message": f"Task execution failed: {e}"}]
