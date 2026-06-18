from ...utils.registry import TASK_REGISTRY
from ..base_task import BaseTask


@TASK_REGISTRY.register("region_detection")
class RegionDetectionTask(BaseTask):
    def __init__(self, model):
        super().__init__(model)

    def predict(self, input_data):
        return self.model.predict(input_data)
