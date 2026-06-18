import os
from ...utils.registry import MODEL_REGISTRY
from paddlex import create_model


@MODEL_REGISTRY.register('region_detection')
class RegionDetectionPP:
    def __init__(self, config, device):
        model_dir = os.path.join(config['ocr_model_weights'], config["model_dir"])
        model_name = config["model_name"]
        self.model = create_model(model_name=model_name, model_dir=model_dir, device=device)

        self.layout_nms = config.get('layout_nms')
        self.layout_merge_bboxes_mode = config.get("layout_merge_bboxes_mode")
        self.visualize = config.get('visualize', False)


    def predict(self, input_data):
        region_det_results = list(
            self.model(
                input_data,
                layout_nms=self.layout_nms,
                layout_merge_bboxes_mode=self.layout_merge_bboxes_mode,
            ),
        )

        if self.visualize:
           pass
        return region_det_results

