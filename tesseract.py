from typing import List, Dict
import os

from PIL import Image, ImageOps

import pytesseract as pt

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse


class ImageRecognition(LabelStudioMLBase):
    def __init__(self,
                 **kwargs) -> None:
        super(ImageRecognition, self).__init__(**kwargs)

        # Task type
        self.task_types = ["OCR", "caption"]
        self.task_type = os.getenv("TASK_TYPE")
        # print(f"Task type is {self.task_type}.")
        # print(f"OCR config is {self.ocr_config}.")

        # From name, to name
        self.from_name = "transcription" if self.task_type == "OCR" else "caption"
        self.to_name = "image"

        # Cache
        self.model_dir = os.environ.get('MODEL_DIR', '.')
        self.cache_dir = os.path.join(self.model_dir, '.file-cache')
        os.makedirs(self.cache_dir, exist_ok=True)

    def setup(self) -> None:
        self.ocr_config = os.environ.get("OCR_CONFIG")
        self.set("model_version", f"tesseract {self.ocr_config}")

    def load_image(self, image_url, task_id) -> Image.Image:
        image_path = self.get_local_path(url=image_url,
                                         cache_dir=self.cache_dir,
                                         task_id=task_id)
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        return image

    @staticmethod
    def _crop_image(image, meta) -> Image:
        # Calculate image properties
        x = meta["x"] * meta["original_width"] / 100
        y = meta["y"] * meta["original_height"] / 100
        w = meta["width"] * meta["original_width"] / 100
        h = meta["height"] * meta["original_height"] / 100

        # Crop image
        image = image.crop((x, y, x + w, y + h))

        return image

    @staticmethod
    def _extract_meta(task) -> Dict:
        meta = dict()
        if task:
            meta['id'] = task['id']
            meta['from_name'] = task['from_name']
            meta['to_name'] = task['to_name']
            meta['type'] = task['type']
            meta['x'] = task['value']['x']
            meta['y'] = task['value']['y']
            meta['width'] = task['value']['width']
            meta['height'] = task['value']['height']
            meta["original_width"] = task['original_width']
            meta["original_height"] = task['original_height']

        return meta

    @staticmethod
    def _fill_temp(meta,
                   from_name,
                   result_text) -> Dict:
        temp = {
            "original_width": meta["original_width"],
            "original_height": meta["original_height"],
            "image_rotation": 0,
            "value": {
                "x": meta['x'],
                "y": meta['y'],
                "width": meta["width"],
                "height": meta["height"],
                "rotation": 0,
                "text": [result_text]},
            "id": meta["id"],
            "from_name": from_name,
            "to_name": meta['to_name'],
            "type": "textarea",
            "origin": "manual"
        }

        return temp

    def predict(self, tasks, **kwargs) -> List:
        if self.task_type == "OCR":
            predictions = self.predict_bbox(tasks, **kwargs)
        else:
            predictions = self.predict_caption(tasks, **kwargs)

        return predictions

    def predict_bbox(self, tasks, **kwargs) -> List:
        # Extract task metadata
        from_name, to_name, value = self.label_interface.get_first_tag_occurence('TextArea',
                                                                                 'Image')
        task = tasks[0]
        img_url = task["data"][value]
        context = kwargs.get('context')

        if context:
            if not context["result"]:
                return []

            image = self.load_image(img_url, task.get('id'))
            result = context.get('result')[-1]

            # Extract meta
            meta = self._extract_meta({**task, **result})
            image = self._crop_image(image, meta)

            # Predict text
            pred_text = pt.image_to_string(
                image, config=self.ocr_config).strip()

            temp = self._fill_temp(meta, from_name, pred_text)

            return [{'result': [temp, result],
                     'score': 0,
                     'model_version': self.get('model_version')}]
        else:
            return []

    def predict_caption(self, tasks, **kwargs) -> ModelResponse:
        # Extract task
        from_name, to_name, value = self.label_interface.get_first_tag_occurence('TextArea',
                                                                                 'Image')
        task = tasks[0]

        # Load image
        img_url = task["data"][value]
        image = self.load_image(img_url, task.get('id'))

        # Recognize text
        pred_text = pt.image_to_string(image, config=self.ocr_config).strip()

        # Fill predictions
        results = [{"from_name": from_name,
                    "to_name": to_name,
                    "type": "textarea",
                    "origin": "manual",
                    "value": {
                        "text": [pred_text]}}]
        predictions = [{"result": results,
                        "model_version": self.model_version}]

        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs) -> None:
        raise NotImplementedError("Training is not implemented yet")
