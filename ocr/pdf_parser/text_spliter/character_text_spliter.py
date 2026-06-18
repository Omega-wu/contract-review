import bisect
import copy
from typing import Any, Iterable, List, Optional, Union, Literal

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class IntervalSearch(object):
    def __init__(self, inters):
        arrs = []
        for inter in inters:
            arrs.extend(inter)
        self.arrs = arrs

    def find(self, inter):
        low_bound1 = bisect.bisect_left(self.arrs, inter[0])
        low_bound2 = bisect.bisect_left(self.arrs, inter[1])
        return [low_bound1 // 2, low_bound2 // 2]


# chunk_size: int = 1000,
# chunk_overlap: int = 0,
class ModifyCharacterTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
            self,
            separators: Optional[List[str]] = ["\n"],
            keep_separator: Union[bool, Literal["start", "end"]] = False,
            strip_whitespace: bool = False,
            is_separator_regex: bool = False,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(
            separators=separators,
            keep_separator=keep_separator,
            is_separator_regex=is_separator_regex,
            strip_whitespace=strip_whitespace,
            **kwargs
        )

    def split_texts(self, docs: list, file_info: dict) -> List[Document]:
        texts, metadatas = [], []
        for doc in docs:
            texts.append(doc[0])
            metadatas.append(doc[1])

        return self.create_documents(texts, metadatas=metadatas, file_info=file_info)

    def create_documents(
            self, texts: List[str], metadatas: Optional[List[dict]] = None, file_info: Optional[dict] = None
    ):
        """Create documents from a list of texts."""

        documents = []
        for i, text in enumerate(texts):
            try:
                file_name = metadatas[i].get('sources', [])[0]
            except:
                file_name = file_info.get('file_name', "")
            index = -1
            indexes = metadatas[i].get('indexes', [])
            pages = metadatas[i].get('pages', [])
            pages_height = metadatas[i].get('page_height', [])
            pages_width = metadatas[i].get('page_width', [])
            bboxes = metadatas[i].get('bboxes', [])
            types = metadatas[i].get('types', [])
            searcher = IntervalSearch(indexes)
            split_texts = self.split_text(text)
            if len(split_texts) == 0:
                split_texts = [text]
            for sort_index, chunk in enumerate(split_texts):
                # new_metadata = copy.deepcopy(metadatas[i])
                new_metadata = {"parent_title": file_name,
                                "multi_title": sort_index,
                                "title": "",
                                "type": types[sort_index],
                                "sort": sort_index,
                                "file_name": file_name,
                                "md5": file_info.get("md5"),
                                "file_id": file_info.get("file_id"),
                                "partition_key": file_info.get("partition_key"),
                                "file_directory": file_info.get("file_directory"),
                                "content": chunk,
                                "content_type": "ocr",
                                "location": []}
                if indexes and bboxes:
                    index = text.find(chunk, index + 1)
                    inter0 = [index, index + len(chunk) - 1]
                    norm_inter = searcher.find(inter0)
                    new_metadata['location'] = []
                    for j in range(norm_inter[0], norm_inter[1] + 1):
                        page = pages[j]
                        page_height = pages_height[j]
                        page_width = pages_width[j]
                        leftTop = {"x": bboxes[j][0], "y": bboxes[j][1]}
                        rightBottom = {"x": bboxes[j][2], "y": bboxes[j][3]}
                        new_metadata["location"].append(
                            {'page': page,
                             'width': page_width,
                             'height': page_height,
                             'leftTop': leftTop,
                             'rightBottom': rightBottom
                             })

                new_doc = dict(page_content=chunk, metadata=new_metadata)
                documents.append(new_doc)
        return documents
