import time
from PIL import Image
from loguru import logger
from ...utils.registry import TASK_REGISTRY
import numpy as np
import pickle
import base64
import requests

import copy
from ...utils.merge import merge_partitions, merge_partitions_ppt

from ...utils.batch_sampler import ImageBatchSampler
from ...utils.reader import ReadImage
from .settings import BLOCK_LABEL_MAP, BLOCK_SETTINGS, REGION_SETTINGS
from .utils import remove_overlap_blocks, convert_formula_res_to_ocr_format, update_region_box, \
    caculate_bbox_area, get_sub_regions_ocr_res, get_bbox_intersection, calculate_overlap_ratio, \
    calculate_minimum_enclosing_bbox, shrink_supplement_region_bbox
from .layout_objects import LayoutBlock, LayoutRegion
from .xycut_enhanced import xycut_enhanced


@TASK_REGISTRY.register("unstructed")
class PDF2TXT:
    def __init__(self, layout_model=None, region_model=None, formula_rec_model=None, ocr_model=None,
                 table_rec_model=None, batch_size=1):
        self.layout_model = layout_model
        self.region_model = region_model
        self.formula_rec_model = formula_rec_model
        self.ocr_model = ocr_model
        self.table_rec_mdoel = table_rec_model
        self.batch_sampler = ImageBatchSampler(batch_size=batch_size)
        self.img_reader = ReadImage(format="BGR")

    def process(self, input_path, file_info={}):
        pdf_start = time.time()
        pdf_res = list(self.predict(input_path, file_info))
        # file_name = file_info.get("file_name")
        # file_name_suffix = file_name.rsplit('.', 1)[1]
        # if file_name_suffix.lower() in ["ppt", "pptx"]:
        #     documents = merge_partitions_ppt(partitions=pdf_res, fpath=input_path, file_info=file_info)
        #     return documents
        txt_content, metadata = merge_partitions(partitions=pdf_res, fpath=input_path, file_info=file_info)
        logger.info(f'pdf cost total time {time.time() - pdf_start}s')
        return txt_content, metadata

    def gather_imgs(self, original_img, layout_det_objs):
        imgs_in_doc = []
        for det_obj in layout_det_objs:
            if det_obj["label"] in BLOCK_LABEL_MAP["image_labels"]:
                label = det_obj["label"]
                x_min, y_min, x_max, y_max = list(map(int, det_obj["coordinate"]))
                img_path = f"imgs/img_in_{label}_box_{x_min}_{y_min}_{x_max}_{y_max}.jpg"
                img = Image.fromarray(original_img[y_min:y_max, x_min:x_max, ::-1])
                imgs_in_doc.append(
                    {
                        "path": img_path,
                        "img": img,
                        "coordinate": (x_min, y_min, x_max, y_max),
                        "score": det_obj["score"],
                    }
                )
        return imgs_in_doc

    def standardized_data(
            self,
            image: list,
            region_det_res,
            layout_det_res,
            overall_ocr_res,
            formula_res_list,
            text_rec_model,
            text_rec_score_thresh=None,
    ):

        matched_ocr_dict = {}
        region_to_block_map = {}
        block_to_ocr_map = {}
        object_boxes = []
        footnote_list = []
        paragraph_title_list = []
        bottom_text_y_max = 0
        max_block_area = 0.0
        doc_title_num = 0

        base_region_bbox = [65535, 65535, 0, 0]
        layout_det_res = remove_overlap_blocks(
            layout_det_res,
            threshold=0.5,
            smaller=True,
        )

        # convert formula_res_list to OCRResult format
        convert_formula_res_to_ocr_format(formula_res_list, overall_ocr_res)

        # match layout boxes and ocr boxes and get some information for layout_order_config
        for box_idx, box_info in enumerate(layout_det_res["boxes"]):
            box = box_info["coordinate"]
            label = box_info["label"].lower()
            object_boxes.append(box)
            _, _, _, y2 = box

            # update the region box and max_block_area according to the layout boxes
            base_region_bbox = update_region_box(box, base_region_bbox)
            max_block_area = max(max_block_area, caculate_bbox_area(box))

            # set the label of footnote to text, when it is above the text boxes
            if label == "footnote":
                footnote_list.append(box_idx)
            elif label == "paragraph_title":
                paragraph_title_list.append(box_idx)
            if label == "text":
                bottom_text_y_max = max(y2, bottom_text_y_max)
            if label == "doc_title":
                doc_title_num += 1

            if label not in ["formula", "table", "seal"]:
                _, matched_idxes = get_sub_regions_ocr_res(
                    overall_ocr_res, [box], return_match_idx=True
                )
                block_to_ocr_map[box_idx] = matched_idxes
                for matched_idx in matched_idxes:
                    if matched_ocr_dict.get(matched_idx, None) is None:
                        matched_ocr_dict[matched_idx] = [box_idx]
                    else:
                        matched_ocr_dict[matched_idx].append(box_idx)

        # fix the footnote label
        for footnote_idx in footnote_list:
            if (
                    layout_det_res["boxes"][footnote_idx]["coordinate"][3]
                    < bottom_text_y_max
            ):
                layout_det_res["boxes"][footnote_idx]["label"] = "text"

        # check if there is only one paragraph title and without doc_title
        only_one_paragraph_title = len(paragraph_title_list) == 1 and doc_title_num == 0
        if only_one_paragraph_title:
            paragraph_title_block_area = caculate_bbox_area(
                layout_det_res["boxes"][paragraph_title_list[0]]["coordinate"]
            )
            title_area_max_block_threshold = BLOCK_SETTINGS.get(
                "title_conversion_area_ratio_threshold", 0.3
            )
            if (
                    paragraph_title_block_area
                    > max_block_area * title_area_max_block_threshold
            ):
                layout_det_res["boxes"][paragraph_title_list[0]]["label"] = "doc_title"

        # Replace the OCR information of the hurdles.
        for overall_ocr_idx, layout_box_ids in matched_ocr_dict.items():
            if len(layout_box_ids) > 1:
                matched_no = 0
                overall_ocr_box = copy.deepcopy(
                    overall_ocr_res["rec_boxes"][overall_ocr_idx]
                )
                overall_ocr_dt_poly = copy.deepcopy(
                    overall_ocr_res["dt_polys"][overall_ocr_idx]
                )
                for box_idx in layout_box_ids:
                    layout_box = layout_det_res["boxes"][box_idx]["coordinate"]
                    crop_box = get_bbox_intersection(overall_ocr_box, layout_box)
                    for ocr_idx in block_to_ocr_map[box_idx]:
                        ocr_box = overall_ocr_res["rec_boxes"][ocr_idx]
                        iou = calculate_overlap_ratio(ocr_box, crop_box, "small")
                        if iou > 0.8:
                            overall_ocr_res["rec_texts"][ocr_idx] = ""
                    x1, y1, x2, y2 = [int(i) for i in crop_box]
                    crop_img = np.array(image)[y1:y2, x1:x2]
                    crop_img_rec_res = list(text_rec_model([crop_img]))[0]
                    crop_img_dt_poly = get_bbox_intersection(
                        overall_ocr_dt_poly, layout_box, return_format="poly"
                    )
                    crop_img_rec_score = crop_img_rec_res["rec_score"]
                    crop_img_rec_text = crop_img_rec_res["rec_text"]
                    text_rec_score_thresh = (
                        text_rec_score_thresh
                        if text_rec_score_thresh is not None
                        else (self.ocr_model.model.pipeline.text_rec_score_thresh)
                    )
                    if crop_img_rec_score >= text_rec_score_thresh:
                        matched_no += 1
                        if matched_no == 1:
                            # the first matched ocr be replaced by the first matched layout box
                            overall_ocr_res["dt_polys"][
                                overall_ocr_idx
                            ] = crop_img_dt_poly
                            overall_ocr_res["rec_boxes"][overall_ocr_idx] = crop_box
                            overall_ocr_res["rec_polys"][
                                overall_ocr_idx
                            ] = crop_img_dt_poly
                            overall_ocr_res["rec_scores"][
                                overall_ocr_idx
                            ] = crop_img_rec_score
                            overall_ocr_res["rec_texts"][
                                overall_ocr_idx
                            ] = crop_img_rec_text
                        else:
                            # the other matched ocr be appended to the overall ocr result
                            overall_ocr_res["dt_polys"].append(crop_img_dt_poly)
                            if len(overall_ocr_res["rec_boxes"]) == 0:
                                overall_ocr_res["rec_boxes"] = np.array([crop_box])
                            else:
                                overall_ocr_res["rec_boxes"] = np.vstack(
                                    (overall_ocr_res["rec_boxes"], crop_box)
                                )
                            overall_ocr_res["rec_polys"].append(crop_img_dt_poly)
                            overall_ocr_res["rec_scores"].append(crop_img_rec_score)
                            overall_ocr_res["rec_texts"].append(crop_img_rec_text)
                            overall_ocr_res["rec_labels"].append("text")
                            block_to_ocr_map[box_idx].remove(overall_ocr_idx)
                            block_to_ocr_map[box_idx].append(
                                len(overall_ocr_res["rec_texts"]) - 1
                            )

        # use layout bbox to do ocr recognition when there is no matched ocr
        for layout_box_idx, overall_ocr_idxes in block_to_ocr_map.items():
            has_text = False
            for idx in overall_ocr_idxes:
                if overall_ocr_res["rec_texts"][idx] != "":
                    has_text = True
                    break
            if not has_text and layout_det_res["boxes"][layout_box_idx][
                "label"
            ] not in BLOCK_LABEL_MAP.get("vision_labels", []):
                crop_box = layout_det_res["boxes"][layout_box_idx]["coordinate"]
                x1, y1, x2, y2 = [int(i) for i in crop_box]
                crop_img = np.array(image)[y1:y2, x1:x2]
                crop_img_rec_res = list(text_rec_model([crop_img]))[0]
                crop_img_dt_poly = get_bbox_intersection(
                    crop_box, crop_box, return_format="poly"
                )
                crop_img_rec_score = crop_img_rec_res["rec_score"]
                crop_img_rec_text = crop_img_rec_res["rec_text"]
                text_rec_score_thresh = (
                    text_rec_score_thresh
                    if text_rec_score_thresh is not None
                    else (self.ocr_model.model.pipeline.text_rec_score_thresh)
                )
                if crop_img_rec_score >= text_rec_score_thresh:
                    if len(overall_ocr_res["rec_boxes"]) == 0:
                        overall_ocr_res["rec_boxes"] = np.array([crop_box])
                    else:
                        overall_ocr_res["rec_boxes"] = np.vstack(
                            (overall_ocr_res["rec_boxes"], crop_box)
                        )
                    overall_ocr_res["rec_polys"].append(crop_img_dt_poly)
                    overall_ocr_res["rec_scores"].append(crop_img_rec_score)
                    overall_ocr_res["rec_texts"].append(crop_img_rec_text)
                    overall_ocr_res["rec_labels"].append("text")
                    block_to_ocr_map[layout_box_idx].append(
                        len(overall_ocr_res["rec_texts"]) - 1
                    )

        if len(layout_det_res["boxes"]) == 0 and len(overall_ocr_res["rec_boxes"]) > 0:
            for idx, ocr_rec_box in enumerate(overall_ocr_res["rec_boxes"]):
                base_region_bbox = update_region_box(ocr_rec_box, base_region_bbox)
                layout_det_res["boxes"].append(
                    {
                        "label": "text",
                        "coordinate": ocr_rec_box,
                        "score": overall_ocr_res["rec_scores"][idx],
                    }
                )
                block_to_ocr_map[idx] = [idx]

        mask_labels = (
                BLOCK_LABEL_MAP.get("unordered_labels", [])
                + BLOCK_LABEL_MAP.get("header_labels", [])
                + BLOCK_LABEL_MAP.get("footer_labels", [])
        )
        block_bboxes = [box["coordinate"] for box in layout_det_res["boxes"]]
        region_det_res["boxes"] = sorted(
            region_det_res["boxes"],
            key=lambda item: caculate_bbox_area(item["coordinate"]),
        )
        if len(region_det_res["boxes"]) == 0:
            region_det_res["boxes"] = [
                {
                    "coordinate": base_region_bbox,
                    "label": "SupplementaryRegion",
                    "score": 1,
                }
            ]
            region_to_block_map[0] = range(len(block_bboxes))
        else:
            block_idxes_set = set(range(len(block_bboxes)))
            # match block to region
            for region_idx, region_info in enumerate(region_det_res["boxes"]):
                matched_idxes = []
                region_to_block_map[region_idx] = []
                region_bbox = region_info["coordinate"]
                for block_idx in block_idxes_set:
                    if layout_det_res["boxes"][block_idx]["label"] in mask_labels:
                        continue
                    overlap_ratio = calculate_overlap_ratio(
                        region_bbox, block_bboxes[block_idx], mode="small"
                    )
                    if overlap_ratio > REGION_SETTINGS.get(
                            "match_block_overlap_ratio_threshold", 0.8
                    ):
                        matched_idxes.append(block_idx)
                old_region_bbox_matched_idxes = []
                if len(matched_idxes) > 0:
                    while len(old_region_bbox_matched_idxes) != len(matched_idxes):
                        old_region_bbox_matched_idxes = copy.deepcopy(matched_idxes)
                        matched_idxes = []
                        matched_bboxes = [
                            block_bboxes[idx] for idx in old_region_bbox_matched_idxes
                        ]
                        new_region_bbox = calculate_minimum_enclosing_bbox(
                            matched_bboxes
                        )
                        for block_idx in block_idxes_set:
                            if (
                                    layout_det_res["boxes"][block_idx]["label"]
                                    in mask_labels
                            ):
                                continue
                            overlap_ratio = calculate_overlap_ratio(
                                new_region_bbox, block_bboxes[block_idx], mode="small"
                            )
                            if overlap_ratio > REGION_SETTINGS.get(
                                    "match_block_overlap_ratio_threshold", 0.8
                            ):
                                matched_idxes.append(block_idx)
                    for block_idx in matched_idxes:
                        block_idxes_set.remove(block_idx)
                    region_to_block_map[region_idx] = matched_idxes
                    region_det_res["boxes"][region_idx]["coordinate"] = new_region_bbox
            # Supplement region when there is no matched block
            while len(block_idxes_set) > 0:
                unmatched_bboxes = [block_bboxes[idx] for idx in block_idxes_set]
                if len(unmatched_bboxes) == 0:
                    break
                supplement_region_bbox = calculate_minimum_enclosing_bbox(
                    unmatched_bboxes
                )
                matched_idxes = []
                # check if the new region bbox is overlapped with other region bbox, if have, then shrink the new region bbox
                for region_idx, region_info in enumerate(region_det_res["boxes"]):
                    if len(region_to_block_map[region_idx]) == 0:
                        continue
                    region_bbox = region_info["coordinate"]
                    overlap_ratio = calculate_overlap_ratio(
                        supplement_region_bbox, region_bbox
                    )
                    if overlap_ratio > 0:
                        supplement_region_bbox, matched_idxes = (
                            shrink_supplement_region_bbox(
                                supplement_region_bbox,
                                region_bbox,
                                image.shape[1],
                                image.shape[0],
                                block_idxes_set,
                                block_bboxes,
                            )
                        )

                matched_idxes = [
                    idx
                    for idx in matched_idxes
                    if layout_det_res["boxes"][idx]["label"] not in mask_labels
                ]
                if len(matched_idxes) == 0:
                    matched_idxes = [
                        idx
                        for idx in block_idxes_set
                        if layout_det_res["boxes"][idx]["label"] not in mask_labels
                    ]
                    if len(matched_idxes) == 0:
                        break
                matched_bboxes = [block_bboxes[idx] for idx in matched_idxes]
                supplement_region_bbox = calculate_minimum_enclosing_bbox(
                    matched_bboxes
                )
                region_idx = len(region_det_res["boxes"])
                region_to_block_map[region_idx] = list(matched_idxes)
                for block_idx in matched_idxes:
                    block_idxes_set.remove(block_idx)
                region_det_res["boxes"].append(
                    {
                        "coordinate": supplement_region_bbox,
                        "label": "SupplementaryRegion",
                        "score": 1,
                    }
                )

            mask_idxes = [
                idx
                for idx in range(len(layout_det_res["boxes"]))
                if layout_det_res["boxes"][idx]["label"] in mask_labels
            ]
            for idx in mask_idxes:
                bbox = layout_det_res["boxes"][idx]["coordinate"]
                region_idx = len(region_det_res["boxes"])
                region_to_block_map[region_idx] = [idx]
                region_det_res["boxes"].append(
                    {
                        "coordinate": bbox,
                        "label": "SupplementaryRegion",
                        "score": 1,
                    }
                )

        region_block_ocr_idx_map = dict(
            region_to_block_map=region_to_block_map,
            block_to_ocr_map=block_to_ocr_map,
        )

        return region_block_ocr_idx_map, region_det_res, layout_det_res

    def get_layout_parsing_objects(
            self,
            image,
            region_block_ocr_idx_map,
            region_det_res,
            overall_ocr_res,
            layout_det_res,
            table_res_list,
            seal_res_list,
            chart_res_list,
            text_rec_model,
            text_rec_score_thresh=None,
    ):

        table_index = 0
        seal_index = 0
        chart_index = 0
        layout_parsing_blocks = []

        for box_idx, box_info in enumerate(layout_det_res["boxes"]):

            label = box_info["label"]
            block_bbox = box_info["coordinate"]
            rec_res = {"boxes": [], "rec_texts": [], "rec_labels": []}

            block = LayoutBlock(label=label, bbox=block_bbox)

            if label == "table" and len(table_res_list) > 0:
                block.content = table_res_list[table_index]["pred_html"]
                table_index += 1
            elif label == "seal" and len(seal_res_list) > 0:
                block.content = "\n".join(seal_res_list[seal_index]["rec_texts"])
                seal_index += 1
            elif label == "chart" and len(chart_res_list) > 0:
                block.content = chart_res_list[chart_index]
                chart_index += 1
            else:
                if label == "formula":
                    _, ocr_idx_list = get_sub_regions_ocr_res(
                        overall_ocr_res, [block_bbox], return_match_idx=True
                    )
                    region_block_ocr_idx_map["block_to_ocr_map"][box_idx] = ocr_idx_list
                else:
                    ocr_idx_list = region_block_ocr_idx_map["block_to_ocr_map"].get(
                        box_idx, []
                    )
                for box_no in ocr_idx_list:
                    rec_res["boxes"].append(overall_ocr_res["rec_boxes"][box_no])
                    rec_res["rec_texts"].append(
                        overall_ocr_res["rec_texts"][box_no],
                    )
                    rec_res["rec_labels"].append(
                        overall_ocr_res["rec_labels"][box_no],
                    )
                block.update_text_content(
                    image=image,
                    ocr_rec_res=rec_res,
                    text_rec_model=text_rec_model,
                    text_rec_score_thresh=text_rec_score_thresh,
                )

            if (
                    label
                    in ["seal", "table", "formula", "chart"]
                    + BLOCK_LABEL_MAP["image_labels"]
            ):
                x_min, y_min, x_max, y_max = list(map(int, block_bbox))
                img_path = (
                    f"imgs/img_in_{block.label}_box_{x_min}_{y_min}_{x_max}_{y_max}.jpg"
                )
                img = Image.fromarray(image[y_min:y_max, x_min:x_max, ::-1])
                block.image = {"path": img_path, "img": img}

            layout_parsing_blocks.append(block)

        page_region_bbox = [65535, 65535, 0, 0]
        layout_parsing_regions = []
        for region_idx, region_info in enumerate(region_det_res["boxes"]):
            region_bbox = np.array(region_info["coordinate"]).astype("int")
            region_blocks = [
                layout_parsing_blocks[idx]
                for idx in region_block_ocr_idx_map["region_to_block_map"][region_idx]
            ]
            if region_blocks:
                page_region_bbox = update_region_box(region_bbox, page_region_bbox)
                region = LayoutRegion(bbox=region_bbox, blocks=region_blocks)
                layout_parsing_regions.append(region)

        layout_parsing_page = LayoutRegion(
            bbox=np.array(page_region_bbox).astype("int"), blocks=layout_parsing_regions
        )

        return layout_parsing_page

    def sort_layout_parsing_blocks(
            self, layout_parsing_page: LayoutRegion
    ):
        layout_parsing_regions = xycut_enhanced(layout_parsing_page)
        parsing_res_list = []
        for region in layout_parsing_regions:
            layout_parsing_blocks = xycut_enhanced(region)
            parsing_res_list.extend(layout_parsing_blocks)

        return parsing_res_list

    def get_layout_parsing_res(
            self,
            image,
            region_det_res,
            layout_det_res,
            overall_ocr_res,
            table_res_list,
            seal_res_list,
            chart_res_list,
            formula_res_list,
            text_rec_score_thresh=None
    ) -> list:

        # Standardize data
        region_block_ocr_idx_map, region_det_res, layout_det_res = (
            self.standardized_data(
                image=image,
                region_det_res=region_det_res,
                layout_det_res=layout_det_res,
                overall_ocr_res=overall_ocr_res,
                formula_res_list=formula_res_list,
                text_rec_model=self.ocr_model.model.pipeline.text_rec_model,
                text_rec_score_thresh=text_rec_score_thresh,
            )
        )

        # Format layout parsing block
        layout_parsing_page = self.get_layout_parsing_objects(
            image=image,
            region_block_ocr_idx_map=region_block_ocr_idx_map,
            region_det_res=region_det_res,
            overall_ocr_res=overall_ocr_res,
            layout_det_res=layout_det_res,
            table_res_list=table_res_list,
            seal_res_list=seal_res_list,
            chart_res_list=chart_res_list,
            text_rec_model=self.ocr_model.model.pipeline.text_rec_model,
            text_rec_score_thresh=self.ocr_model.model.pipeline.text_rec_score_thresh,
        )

        parsing_res_list = self.sort_layout_parsing_blocks(layout_parsing_page)

        index = 1
        for block in parsing_res_list:
            if block.label in BLOCK_LABEL_MAP["visualize_index_labels"]:
                block.order_index = index
                index += 1

        return parsing_res_list

    def layout_pdf_service(self, doc_preprocessor_images, url):
        array_pickled = pickle.dumps(doc_preprocessor_images)
        array_base64 = base64.b64encode(array_pickled).decode('utf-8')

        # 发送 POST 请求
        data = {
            'array_pickled': array_base64
        }

        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()

        return result

    def predict(
            self,
            input,
            file_info,
            text_rec_score_thresh=None,
    ):
        for batch_data in self.batch_sampler(input):
            image_arrays = self.img_reader(batch_data.instances)
            doc_preprocessor_results = [{"output_img": arr} for arr in image_arrays]
            doc_preprocessor_images = [
                item["output_img"] for item in doc_preprocessor_results
            ]
            # print(doc_preprocessor_images)
            if file_info.get("layout_url"):
                layout_det_results = self.layout_pdf_service(doc_preprocessor_images, file_info.get("layout_url"))
            else:
                layout_det_results = self.layout_model.predict(doc_preprocessor_images)

            for i in layout_det_results:
                for j in i["boxes"]:
                    if j["label"] == "table" and j["cls_id"] == 8:
                        j["label"] = "abstract"
                        j["cls_id"] = 4

            print(layout_det_results)
            print(len(layout_det_results))
            imgs_in_doc = [
                self.gather_imgs(img, res["boxes"])
                for img, res in zip(doc_preprocessor_images, layout_det_results)
            ]

            if self.region_model:
                region_det_results = self.region_model.predict(doc_preprocessor_images)
                # print(region_det_results)
            else:
                region_det_results = [{"boxes": []} for _ in doc_preprocessor_images]

            if self.formula_rec_model:
                formula_res_all = self.formula_rec_model.predict(doc_preprocessor_images, layout_det_results)

                formula_res_lists = [
                    item["formula_res_list"] for item in formula_res_all
                ]
                # print(formula_res_lists)
            else:
                formula_res_lists = [[] for _ in doc_preprocessor_images]

            for doc_preprocessor_image, formula_res_list in zip(
                    doc_preprocessor_images, formula_res_lists
            ):
                for formula_res in formula_res_list:
                    x_min, y_min, x_max, y_max = list(map(int, formula_res["dt_polys"]))
                    doc_preprocessor_image[y_min:y_max, x_min:x_max, :] = 255.0

            overall_ocr_results = self.ocr_model.predict(doc_preprocessor_images)
            # print(overall_ocr_results)
            for overall_ocr_res in overall_ocr_results:
                overall_ocr_res["rec_labels"] = ["text"] * len(
                    overall_ocr_res["rec_texts"]
                )

            if self.table_rec_mdoel:
                table_res_lists = []
                for (
                        layout_det_res,
                        doc_preprocessor_image,
                        overall_ocr_res,
                        formula_res_list,
                        imgs_in_doc_for_img,
                ) in zip(
                    layout_det_results,
                    doc_preprocessor_images,
                    overall_ocr_results,
                    formula_res_lists,
                    imgs_in_doc,
                ):
                    table_contents_for_img = copy.deepcopy(overall_ocr_res)
                    for formula_res in formula_res_list:
                        x_min, y_min, x_max, y_max = list(
                            map(int, formula_res["dt_polys"])
                        )
                        poly_points = [
                            (x_min, y_min),
                            (x_max, y_min),
                            (x_max, y_max),
                            (x_min, y_max),
                        ]
                        table_contents_for_img["dt_polys"].append(poly_points)
                        rec_formula = formula_res["rec_formula"]
                        if not rec_formula.startswith("$") or not rec_formula.endswith(
                                "$"
                        ):
                            rec_formula = f"${rec_formula}$"
                        table_contents_for_img["rec_texts"].append(f"{rec_formula}")
                        if table_contents_for_img["rec_boxes"].size == 0:
                            table_contents_for_img["rec_boxes"] = np.array(
                                [formula_res["dt_polys"]]
                            )
                        else:
                            table_contents_for_img["rec_boxes"] = np.vstack(
                                (
                                    table_contents_for_img["rec_boxes"],
                                    [formula_res["dt_polys"]],
                                )
                            )
                        table_contents_for_img["rec_polys"].append(poly_points)
                        table_contents_for_img["rec_scores"].append(1)

                    for img in imgs_in_doc_for_img:
                        img_path = img["path"]
                        x_min, y_min, x_max, y_max = img["coordinate"]
                        poly_points = [
                            (x_min, y_min),
                            (x_max, y_min),
                            (x_max, y_max),
                            (x_min, y_max),
                        ]
                        table_contents_for_img["dt_polys"].append(poly_points)
                        table_contents_for_img["rec_texts"].append(
                            f'<div style="text-align: center;"><img src="{img_path}" alt="Image" /></div>'
                        )
                        if table_contents_for_img["rec_boxes"].size == 0:
                            table_contents_for_img["rec_boxes"] = np.array(
                                [img["coordinate"]]
                            )
                        else:
                            table_contents_for_img["rec_boxes"] = np.vstack(
                                (table_contents_for_img["rec_boxes"], img["coordinate"])
                            )
                        table_contents_for_img["rec_polys"].append(poly_points)
                        table_contents_for_img["rec_scores"].append(img["score"])

                    table_res_all = self.table_rec_mdoel.predict(doc_preprocessor_image,
                                                                 overall_ocr_res=table_contents_for_img,
                                                                 layout_det_res=layout_det_res)

                    single_table_res_lists = [
                        item["table_res_list"] for item in table_res_all
                    ]
                    table_res_lists.extend(single_table_res_lists)
            else:
                table_res_lists = [[] for _ in doc_preprocessor_images]

            seal_res_lists = [[] for _ in doc_preprocessor_images]

            for (
                    input_path,
                    page_index,
                    doc_preprocessor_image,
                    doc_preprocessor_res,
                    layout_det_res,
                    region_det_res,
                    overall_ocr_res,
                    table_res_list,
                    seal_res_list,
                    formula_res_list,
                    imgs_in_doc_for_img,
            ) in zip(
                batch_data.input_paths,
                batch_data.page_indexes,
                doc_preprocessor_images,
                doc_preprocessor_results,
                layout_det_results,
                region_det_results,
                overall_ocr_results,
                table_res_lists,
                seal_res_lists,
                formula_res_lists,
                imgs_in_doc,
            ):
                chart_res_list = []

                parsing_res_list = self.get_layout_parsing_res(
                    doc_preprocessor_image,
                    region_det_res=region_det_res,
                    layout_det_res=layout_det_res,
                    overall_ocr_res=overall_ocr_res,
                    table_res_list=table_res_list,
                    seal_res_list=seal_res_list,
                    chart_res_list=chart_res_list,
                    formula_res_list=formula_res_list,
                    text_rec_score_thresh=text_rec_score_thresh,
                )

                for formula_res in formula_res_list:
                    x_min, y_min, x_max, y_max = list(map(int, formula_res["dt_polys"]))
                    doc_preprocessor_image[y_min:y_max, x_min:x_max, :] = formula_res[
                        "input_img"
                    ]

                single_img_res = {
                    "input_path": input_path,
                    "page_index": page_index,
                    "doc_preprocessor_res": doc_preprocessor_res,
                    "layout_det_res": layout_det_res,
                    "region_det_res": region_det_res,
                    "overall_ocr_res": overall_ocr_res,
                    "table_res_list": table_res_list,
                    "seal_res_list": seal_res_list,
                    "chart_res_list": chart_res_list,
                    "formula_res_list": formula_res_list,
                    "parsing_res_list": parsing_res_list,
                    "imgs_in_doc": imgs_in_doc_for_img,
                }
                yield single_img_res
