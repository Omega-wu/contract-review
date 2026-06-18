from ...utils.registry import MODEL_REGISTRY
from paddlex import create_pipeline
import os


@MODEL_REGISTRY.register('table_recognition_sla')
class PPTableSLAModel(object):

    def __init__(self, config, device):
        config["SubModules"]["TableClassification"]["model_dir"] = os.path.join(
            config["ocr_model_weights"],
            config["SubModules"][
                "TableClassification"][
                "model_dir"])
        config["SubModules"]["WiredTableStructureRecognition"]["model_dir"] = os.path.join(
            config["ocr_model_weights"],
            config["SubModules"][
                "WiredTableStructureRecognition"][
                "model_dir"])
        config["SubModules"]["WirelessTableStructureRecognition"]["model_dir"] = os.path.join(
            config["ocr_model_weights"],
            config["SubModules"][
                "WirelessTableStructureRecognition"][
                "model_dir"])
        config["SubModules"]["WiredTableCellsDetection"]["model_dir"] = os.path.join(
            config["ocr_model_weights"],
            config["SubModules"][
                "WiredTableCellsDetection"][
                "model_dir"])
        config["SubModules"]["WirelessTableCellsDetection"]["model_dir"] = os.path.join(
            config["ocr_model_weights"],
            config["SubModules"][
                "WirelessTableCellsDetection"][
                "model_dir"])
        config["SubModules"]["TableOrientationClassify"]["model_dir"] = os.path.join(
            config["ocr_model_weights"],
            config["SubModules"][
                "TableOrientationClassify"][
                "model_dir"])

        config["SubPipelines"]["GeneralOCR"]["SubModules"]["TextDetection"]["model_dir"] = os.path.join(
            config["ocr_model_weights"],
            config["SubPipelines"]["GeneralOCR"]["SubModules"]["TextDetection"]["model_dir"])
        config["SubPipelines"]["GeneralOCR"]["SubModules"]["TextLineOrientation"]["model_dir"] = os.path.join(
            config["ocr_model_weights"],
            config["SubPipelines"]["GeneralOCR"]["SubModules"]["TextLineOrientation"]["model_dir"])
        config["SubPipelines"]["GeneralOCR"]["SubModules"]["TextRecognition"]["model_dir"] = os.path.join(
            config["ocr_model_weights"],
            config["SubPipelines"]["GeneralOCR"]["SubModules"]["TextRecognition"]["model_dir"])





        self.pipeline = create_pipeline(config=config, device=device)

    def predict(self, input_data, overall_ocr_res=None,
                layout_det_res=None):
        table_res_all = list(
            self.pipeline(
                input_data,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_layout_detection=False,
                use_ocr_model=False,
                overall_ocr_res=overall_ocr_res,
                layout_det_res=layout_det_res,
                cell_sort_by_y_projection=True,
                use_wired_table_cells_trans_to_html=False,
                use_wireless_table_cells_trans_to_html=False,
                use_table_orientation_classify=True,
                use_ocr_results_with_table_cells=True,
                use_e2e_wired_table_rec_model=False,
                use_e2e_wireless_table_rec_model=True,
            ),
        )

        return table_res_all
