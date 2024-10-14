from .base_solver import Solver
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from PIL import Image


class MPLUGSolver(Solver):
    def __init__(self, image_root, debug_mode, **config):
        super().__init__(image_root, debug_mode)
        self.solver_name = config.get("name", "mplug_owl3")
        self.config = config
        huggingface_model_id = config.get(
            "huggingface_model_id", "mPLUG/mPLUG-Owl3-7B-240728"
        )
        # TODO: mPLUG didn't actually support multi-gpu by simply setting device_map="auto"
        # We need to figure out a way to make it work
        self.model = AutoModelForCausalLM.from_pretrained(
            huggingface_model_id,
            torch_dtype=torch.half,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(huggingface_model_id)
        self.processor = self.model.init_processor(self.tokenizer)

    @torch.inference_mode()
    def generate(self, prompt, image_paths):
        images = []
        for image_path in image_paths:
            images.append(Image.open(image_path).convert("RGB"))
        real_prompt = "".join(["<|image|>"] * len(image_paths)) + prompt
        messages = [
            {"role": "user", "content": real_prompt},
            {"role": "assistant", "content": ""},
        ]
        inputs = self.processor(messages, images=images, video=None).to("cuda")
        generated_text = self.model.generate(
            **inputs, tokenizer=self.tokenizer, max_new_tokens=512, decode_text=True
        )[0]
        return generated_text, None
