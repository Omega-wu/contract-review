import os
from ...utils.registry import MODEL_REGISTRY
from paddlex import create_model
from loguru import logger

@MODEL_REGISTRY.register('layout_detection_pp')
class LayoutDetectionPP:

    def __init__(self, config, device):

        model_dir = os.path.join(config['ocr_model_weights'], config["model_dir"])
        model_name = config["model_name"]
        self.model = create_model(model_name=model_name, model_dir=model_dir, device=device)

        self.threshold = config.get('threshold')
        self.layout_nms = config.get('layout_nms')
        self.layout_unclip_ratio = config.get('layout_unclip_ratio')
        self.layout_merge_bboxes_mode = config.get('layout_merge_bboxes_mode')
        self.batch_size = config.get("batch_size")
        self.visualize = config.get('visualize', False)

    def predict(self, input_data):
        layout_det_results = list(
            self.model(
                input_data,
                threshold=self.threshold,
                layout_nms=self.layout_nms,
                layout_unclip_ratio=self.layout_unclip_ratio,
                layout_merge_bboxes_mode=self.layout_merge_bboxes_mode,
            )
        )

        if 1:
            for res in layout_det_results:
                res.print()
                res.save_to_img(save_path="./output/")
        layout_det_results = [res._to_json()["res"] for res in layout_det_results]
        return layout_det_results

