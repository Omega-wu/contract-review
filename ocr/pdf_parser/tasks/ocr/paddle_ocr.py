import os

from paddlex import create_pipeline
from ...utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register('ocr_ppocr')
class ModifiedPaddleOCR:
    def __init__(self, config, device):
        config["SubModules"]["TextDetection"]["model_dir"] = os.path.join(config["ocr_model_weights"],
                                                                          config["SubModules"]["TextDetection"][
                                                                              "model_dir"])
        # config["SubModules"]["TextLineOrientation"]["model_dir"] = os.path.join(config["ocr_model_weights"],
        #                                                                   config["SubModules"]["TextLineOrientation"][
        #                                                                       "model_dir"])
        config["SubModules"]["TextRecognition"]["model_dir"] = os.path.join(config["ocr_model_weights"],
                                                                                config["SubModules"][
                                                                                    "TextRecognition"][
                                                                                    "model_dir"])
        self.pipeline = create_pipeline(config=config, device=device)

    def predict(self, input_data):
        overall_ocr_results = list(
            self.pipeline(
                input_data
            ),
        )

        return overall_ocr_results
