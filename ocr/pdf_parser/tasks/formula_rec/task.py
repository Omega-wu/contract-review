from ...utils.registry import TASK_REGISTRY
from ..base_task import BaseTask


@TASK_REGISTRY.register("formula_recognition")
class FormulaRecognitionTask(BaseTask):
    def __init__(self, model):
        super().__init__(model)

    def predict(self, input_data, layout_det_results=None):

        return self.model.predict(input_data, layout_det_results)