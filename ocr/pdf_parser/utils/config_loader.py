import yaml
from loguru import logger

from .registry import TASK_REGISTRY, MODEL_REGISTRY


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def initialize_tasks_and_models(config):
    tasks = {}
    device = config.get("device", "cpu")
    for task_name in config['tasks']:
        model_name = config['tasks'][task_name]['model']
        model_config = config['tasks'][task_name]['model_config']
        model_config["ocr_model_weights"] = config["ocr_model_weights"]

        task_cls = TASK_REGISTRY.get(task_name)
        model_cls = MODEL_REGISTRY.get(model_name)
        logger.info(f"Initializing {task_name} task with {model_cls}")

        model_obj = model_cls(model_config, device)
        task_obj = task_cls(model_obj)
        tasks[task_name] = task_obj

    return tasks
