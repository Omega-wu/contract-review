import pypdfium2 as pdfium
import cv2
import numpy as np

class _BaseReader(object):
    """_BaseReader"""

    def __init__(self, backend, **bk_args):
        super().__init__()
        if len(bk_args) == 0:
            bk_args = self.get_default_backend_args()
        self.bk_type = backend
        self.bk_args = bk_args
        self._backend = self.get_backend()

    def read(self, in_path):
        """read file from path"""
        raise NotImplementedError

    def get_backend(self, bk_args=None):
        """get the backend"""
        if bk_args is None:
            bk_args = self.bk_args
        return self._init_backend(self.bk_type, bk_args)

    def set_backend(self, backend, **bk_args):
        self.bk_type = backend
        self.bk_args = bk_args
        self._backend = self.get_backend()

    def _init_backend(self, bk_type, bk_args):
        """init backend"""
        raise NotImplementedError


    def get_default_backend_args(self):
        """get default backend arguments"""
        return {}


class _BaseReaderBackend(object):
    """_BaseReaderBackend"""

    def read_file(self, in_path):
        """read file from path"""
        raise NotImplementedError

class OpenCVImageReaderBackend(_BaseReaderBackend):
    """OpenCVImageReaderBackend"""

    def __init__(self, flags=None):
        super().__init__()
        if flags is None:
            flags = cv2.IMREAD_COLOR
        self.flags = flags

    def read_file(self, in_path):
        """read image file from path by OpenCV"""
        return cv2.imread(in_path, flags=self.flags)

class ImageReader(_BaseReader):
    """ImageReader"""

    def __init__(self, backend="opencv", **bk_args):
        super().__init__(backend=backend, **bk_args)

    def read(self, in_path):
        """read the image file from path"""
        arr = self._backend.read_file(str(in_path))
        return arr

    def _init_backend(self, bk_type, bk_args):
        """init backend"""
        if bk_type == "opencv":
            return OpenCVImageReaderBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")


class ReadImage:
    """Load image from the file."""

    def __init__(self, format="BGR"):
        """
        Initialize the instance.

        Args:
            format (str, optional): Target color format to convert the image to.
                Choices are 'BGR', 'RGB', and 'GRAY'. Default: 'BGR'.
        """
        super().__init__()
        self.format = format
        flags = {
            "BGR": cv2.IMREAD_COLOR,
            "RGB": cv2.IMREAD_COLOR,
            "GRAY": cv2.IMREAD_GRAYSCALE,
        }[self.format]
        self._img_reader = ImageReader(backend="opencv", flags=flags)

    def __call__(self, imgs):
        """apply"""
        return [self.read(img) for img in imgs]

    def read(self, img):
        if isinstance(img, np.ndarray):
            if self.format == "RGB":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif isinstance(img, str):
            blob = self._img_reader.read(img)
            if blob is None:
                raise Exception(f"Image read Error: {img}")

            if self.format == "RGB":
                if blob.ndim != 3:
                    raise RuntimeError("Array is not 3-dimensional.")
                # BGR to RGB
                blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
            return blob
        else:
            raise TypeError(
                f"ReadImage only supports the following types:\n"
                f"1. str, indicating a image file path or a directory containing image files.\n"
                f"2. numpy.ndarray.\n"
                f"However, got type: {type(img).__name__}."
            )

class PDFReaderBackend(_BaseReaderBackend):

    def __init__(self, rotate=0, zoom=2.0):
        super().__init__()
        self._rotation = rotate
        self._scale = zoom

    def read_file(self, in_path):
        doc = pdfium.PdfDocument(in_path)
        try:
            for page in doc:
                image = page.render(scale=self._scale, rotation=self._rotation).to_pil()
                image = image.convert("RGB")
                img_cv = np.array(image)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                yield img_cv
        finally:
            doc.close()


class PDFReader(_BaseReader):
    """PDFReader"""

    def __init__(self, backend="pypdfium2", **bk_args):
        super().__init__(backend, **bk_args)

    def read(self, in_path):
        yield from self._backend.read_file(str(in_path))

    def _init_backend(self, bk_type, bk_args):
        return PDFReaderBackend(**bk_args)

