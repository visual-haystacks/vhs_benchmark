from .base_solver import Solver
from transformers import Idefics2Processor, Idefics2ForConditionalGeneration
import torch
from PIL import Image


class IDEFICS2Solver(Solver):
    def __init__(self, image_root, debug_mode, **config):
        super().__init__(image_root, debug_mode)
        self.solver_name = config.get("name", "Idefics2")
        self.config = config
        huggingface_model_id = config.get(
            "huggingface_model_id", "HuggingFaceM4/idefics2-8b"
        )
        self.processor = Idefics2Processor.from_pretrained(huggingface_model_id)
        self.model = Idefics2ForConditionalGeneration.from_pretrained(
            huggingface_model_id,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate(self, prompt, image_paths):
        content = [{"type": "text", "text": prompt}]
        images = []
        # Idefics2 (https://arxiv.org/abs/2405.02246) handles at most 32 images
        # 2'048 context length / 64 tokens per image = 32 images
        image_paths = image_paths[:32]
        for idx, image_path in enumerate(image_paths):
            content.append({"type": "image"})
            images.append(Image.open(image_path))
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
