from ...utils.registry import TASK_REGISTRY
from ..base_task import  BaseTask


@TASK_REGISTRY.register("table_recognition")
class TableRecognitionTaskTask(BaseTask):
    def __init__(self, model):
        super().__init__(model)

    def predict(self, input_data, overall_ocr_res=None, layout_det_res=None):
        return self.model.predict(input_data,  overall_ocr_res, layout_det_res)

