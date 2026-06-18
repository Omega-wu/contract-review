import os
from paddlex import create_pipeline
from ...utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register('formula_recognition_pp')
class FormulaRecognitionPP:
    def __init__(self, config, device):
        config["SubModules"]["FormulaRecognition"]["model_dir"] = os.path.join(config["ocr_model_weights"],
                                                                               config["SubModules"]["FormulaRecognition"]["model_dir"])
        self.pipeline = create_pipeline(config=config, device=device)

    def predict(self, input_data, layout_det_results=None):
        formula_res_all = list(
            self.pipeline(
                input_data,
                use_layout_detection=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                layout_det_res=layout_det_results,
            ),
        )

        return formula_res_all
