from .base_solver import Solver
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image


class LLaVASolver(Solver):
    def __init__(self, image_root, debug_mode, **config):
        super().__init__(image_root, debug_mode)
        self.solver_name = config.get("name", "LLaVA-v1.5-7b")
        self.config = config
        huggingface_model_id = config.get(
            "huggingface_model_id", "llava-hf/llava-1.5-7b-hf"
        )
        self.processor = AutoProcessor.from_pretrained(huggingface_model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            huggingface_model_id,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate(self, prompt, image_paths):
        image_paths = image_paths[:3]  # LLAVA handles at most 3 images
        image_tokens = "<image>\n" * len(image_paths)
        question = f"USER: {image_tokens}{prompt}\nASSISTANT:"
        images = []
        for idx, image_path in enumerate(image_paths):
            images.append(Image.open(image_path))
        inputs = self.processor(text=question, images=images, return_tensors="pt").to(
            self.device
        )

        generated_id = self.model.generate(
            **inputs, max_new_tokens=500, do_sample=False
        )
        generated_text = self.processor.batch_decode(
            generated_id, skip_special_tokens=True
        )[0]
        # Hack: remove the prefix question
        generated_text = generated_text.split("ASSISTANT:")[1]
        return generated_text, None
