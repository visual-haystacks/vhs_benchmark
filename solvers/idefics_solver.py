from .base_solver import Solver
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image


class IDEFICSSolver(Solver):
    def __init__(self, image_root, debug_mode, **config):
        super().__init__(image_root, debug_mode)
        self.solver_name = config.get("name", "Idefics3")
        self.config = config
        huggingface_model_id = config.get(
            "huggingface_model_id", "HuggingFaceM4/Idefics3-8B-Llama3"
        )
        self.processor = AutoProcessor.from_pretrained(huggingface_model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            huggingface_model_id, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        # self.device = "cuda"
        # Get the right self.device
        # ref: https://discuss.pytorch.org/t/automatically-cast-input-to-huggingface-model-s-device-map/198704
        first_layer_name = list(self.model.hf_device_map.keys())[0]
        self.device = self.model.hf_device_map[first_layer_name]

    def generate(self, prompt, image_paths):
        content = [{"type": "text", "text": prompt}]
        images = []
        for idx, image_path in enumerate(image_paths):
            content.append({"type": "image"})
            images.append(Image.open(image_path).convert("RGB"))
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(images=images, text=text, return_tensors="pt").to(
            self.device
        )

        generated_id = self.model.generate(**inputs, max_new_tokens=500)
        generated_text = self.processor.batch_decode(
            generated_id, skip_special_tokens=True
        )[0]
        # Hack: remove the prefix question
        generated_text = generated_text.split("Assistant:")[1]
        return generated_text, None
