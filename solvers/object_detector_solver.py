from .base_solver import Solver
import re
import torch
import numpy as np
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection


class SingleObjectDetectorSolver(Solver):
    def __init__(self, image_root, debug_mode, **config):
        super().__init__(image_root, debug_mode)
        self.solver_name = config.get("name", "object_detector")
        self.config = config

        huggingface_model_id = config.get(
            "huggingface_model_id", "google/owlv2-base-patch16-ensemble"
        )
        self.processor = Owlv2Processor.from_pretrained(huggingface_model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            huggingface_model_id,
            device_map="auto",
        ).eval()

    def find_object_phrases(self, sentence):
        # Compile a regex pattern that allows for text before 'is there a/an'
        pattern = re.compile(r".*is there (an? [\w\s]+)[?!.]*", re.IGNORECASE)
        match = pattern.search(sentence)

        if match:
            return match.group(1)
        assert False, f"No object phrase found in sentence: {sentence}"

    def parse_needle_object(self, needle_object_str):
        # Define the regex pattern to extract the object name
        pattern = r"for the image with (a|an) (.*?),"

        # Use re.search to find the match
        match = re.search(pattern, needle_object_str, re.IGNORECASE)

        if match:
            # Extract the object name
            article = match.group(1)
            object_name = match.group(2)
            return f"{article} {object_name}"
        else:
            raise ValueError("The needle object string is not in the expected format.")

    @torch.inference_mode()
    def run_detection(self, data_paths, needle_object):
        all_confidences = []

        for data_path in data_paths:
            image = Image.open(data_path).convert("RGB")
            inputs = self.processor(
                text=needle_object, images=image, return_tensors="pt"
            )
            inputs = inputs.to("cuda")
            outputs = self.model(**inputs)

            width, height = image.size

            # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
            target_sizes = torch.Tensor([[height, width]])
            target_sizes = target_sizes.to("cuda")
            # Convert outputs (bounding boxes and class logits) to COCO API
            results = self.processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=0
            )
            highest_conf = results[0]["scores"].max()
            all_confidences.append(highest_conf.item())
        return all_confidences

    def generate(self, prompt, data_paths):
        # parse the prompt
        # format: image of a/an object name
        needle_object = ["image of " + self.parse_needle_object(prompt)]
        target_object = ["image of " + self.find_object_phrases(prompt)]

        detected_results = self.run_detection(data_paths, needle_object, batch_size=1)
        # for single needle case, pick the image with the highest confidence as the needle image
        needle_img = data_paths[detected_results.index(max(detected_results))]
        target_detected_results = self.run_detection(
            [needle_img], target_object, batch_size=1
        )
        if max(target_detected_results) < self.config.get("confidence_threshold", 0.2):
            return "No", None
        else:
            return "Yes", None


class MultiObjectDetectorSolver(Solver):
    def __init__(self, image_root, debug_mode, **config):
        super().__init__(image_root, debug_mode)
        self.solver_name = config.get("name", "object_detector")
        self.config = config

        huggingface_model_id = config.get(
            "huggingface_model_id", "google/owlv2-base-patch16-ensemble"
        )
        self.processor = Owlv2Processor.from_pretrained(huggingface_model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            huggingface_model_id,
            device_map="auto",
        ).eval()

    def find_object_phrases(self, sentence):
        # Compile a regex pattern that allows for text before 'is there a/an'
        pattern = re.compile(r".*of them contain (an? [\w\s]+)[?!.]*", re.IGNORECASE)
        match = pattern.search(sentence)

        if match:
            return match.group(1)
        assert False, f"No object phrase found in sentence: {sentence}"

    def parse_needle_object(self, needle_object_str):
        # Define the regex pattern to extract the object name
        pattern = r"for all images with (a|an) (.*?),"

        # Use re.search to find the match
        match = re.search(pattern, needle_object_str, re.IGNORECASE)

        if match:
            # Extract the object name
            article = match.group(1)
            object_name = match.group(2)
            return f"{article} {object_name}"
        else:
            raise ValueError("The needle object string is not in the expected format.")

    @torch.inference_mode()
    def run_detection(self, data_paths, needle_object):
        all_confidences = []

        for data_path in data_paths:
            image = Image.open(data_path).convert("RGB")
            inputs = self.processor(
                text=needle_object, images=image, return_tensors="pt"
            )
            inputs = inputs.to("cuda")
            outputs = self.model(**inputs)

            width, height = image.size

            # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
            target_sizes = torch.Tensor([[height, width]])
            target_sizes = target_sizes.to("cuda")
            # Convert outputs (bounding boxes and class logits) to COCO API
            results = self.processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=0
            )
            highest_conf = results[0]["scores"].max()
            all_confidences.append(highest_conf.item())
        return all_confidences

    def generate(self, prompt, data_paths):
        # parse the prompt
        # format: image of a/an object name
        needle_object = ["image of " + self.parse_needle_object(prompt)]
        target_object = ["image of " + self.find_object_phrases(prompt)]

        detected_results = self.run_detection(data_paths, needle_object, batch_size=1)
        filtered_indices = [
            i for i, score in enumerate(detected_results) if score > 0.2
        ]
        sorted_indices = sorted(
            filtered_indices, key=lambda i: detected_results[i], reverse=True
        )
        selected_indices = sorted_indices[: max(1, min(5, len(sorted_indices)))]
        selected_images = [data_paths[i] for i in selected_indices]
        # for single needle case, pick the image with the highest confidence as the needle image
        target_detected_results = self.run_detection(
            selected_images, target_object, batch_size=1
        )
        # Case 1: All
        if "all" in prompt.split(",")[1]:
            if np.any(
                np.array(target_detected_results)
                < self.config.get("confidence_threshold", 0.2)
            ):
                return "No", None
            else:
                return "Yes", None
        # Case 2: Any
        else:
            if np.any(
                np.array(target_detected_results)
                > self.config.get("confidence_threshold", 0.2)
            ):
                return "Yes", None
            else:
                return "No", None
