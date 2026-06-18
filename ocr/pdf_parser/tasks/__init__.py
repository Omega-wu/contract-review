from .base_task import BaseTask
from .layout.task import LayoutDetectionTask
from .layout.doclayout import LayoutDetectionPP
from .region_dec.region_dectet import RegionDetectionPP
from .region_dec.task import RegionDetectionTask
from .ocr.task import OCRTask
from .ocr.paddle_ocr import ModifiedPaddleOCR
from .formula_rec.pp_rec import FormulaRecognitionPP
from .formula_rec.task import FormulaRecognitionTask
from .table_rec.pp_SLA_model import PPTableSLAModel
from .table_rec.task import TableRecognitionTaskTask
from .unstructed.unstructed_parser import PDF2TXT

__all__ = [
    "BaseTask",
    "LayoutDetectionTask",
    "LayoutDetectionPP",
    "OCRTask",
    "ModifiedPaddleOCR",
    "FormulaRecognitionPP",
    "FormulaRecognitionTask",
    "TableRecognitionTaskTask",
    "PDF2TXT",
    "PPTableSLAModel",
    "RegionDetectionPP",
    "RegionDetectionTask"
]
