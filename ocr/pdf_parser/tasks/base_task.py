import os
import pypdfium2 as pdfium  # pip install pypdfium2
import numpy as np


class BaseTask:
    def __init__(self, model):
        self.model = model

    def load_images(self, input_data):

        images = []

        if os.path.isdir(input_data):
            files = os.listdir(input_data)
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jepg")):
                    image_path = os.path.join(input_data, file)
                    images.append(image_path)
            images = sorted(images)
        else:
            if input_data.lower().endswith(('.png', '.jpg', '.jpeg')):
                images = [input_data]
            else:
                raise ValueError("Unsupported input data format: {}".format(input_data))

        return images

    def load_pdf_images(self, input_data):
        pdf_images = {}

        if os.path.isdir(input_data):
            files = os.listdir(input_data)

            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(input_data, file)
                    images = self.load_pdf(pdf_path)
                    for i, img in enumerate(images):
                        img_id = f"{os.path.splitext(file)[0]}_page_{i + 1:04d}"
                        pdf_images[img_id] = img
        else:
            if input_data.lower().endswith(".pdf"):
                # If input is a single image file
                images = self.load_pdf(input_data)
                for i, img in enumerate(images):
                    img_id = f"{os.path.splitext(os.path.basename(input_data))[0]}_page_{i + 1:04d}"
                    pdf_images[img_id] = img
            else:
                raise ValueError("Unsupported input data format: {}".format(input_data))

        return pdf_images

    def load_pdf(self, pdf_path, dpi=144):
        images_pil = []
        images_ndarray = []
        pdf = pdfium.PdfDocument(pdf_path)
        for page in pdf:
            image = self.load_pdf_page(page, dpi)
            images_pil.append(image)
            images_ndarray.append(np.array(image))
        return images_pil, images_ndarray

    def load_pdf_page(self, page, dpi):
        scale = dpi / 72.0
        bitmap = page.render(
            scale=scale,
            rotation=0
        )
        pil_image = bitmap.to_pil()

        if pil_image.width > 3000 or pil_image.height > 3000:
            bitmap = page.render(
                scale=1.0,
                rotation=0
            )
            pil_image = bitmap.to_pil()

        return pil_image
