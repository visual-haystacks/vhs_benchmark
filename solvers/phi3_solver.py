import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from .base_solver import Solver


class Phi3Solver(Solver):
    def __init__(self, image_root, debug_mode, **config):
        super().__init__(image_root, debug_mode)
        self.solver_name = config.get("name", "phi3")
        self.config = config
        self.huggingface_model_id = config.get(
            "hub_model_id", "microsoft/Phi-3-vision-128k-instruct"
        )
        self.processor = AutoProcessor.from_pretrained(
            self.huggingface_model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.huggingface_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).eval()

    @torch.inference_mode()
    def generate(self, prompt, image_paths):
        # Phi3's template
        # Based on this one: https://huggingface.co/microsoft/Phi-3-vision-128k-instruct
        user_prompt = "<|user|>\n"
        assistant_prompt = "<|assistant|>\n"
        prompt_suffix = "<|end|>\n"
        images = []
        image_tokens = ""
        for img_num, image_path in enumerate(image_paths):
            image_tokens += f"<|image_{img_num+1}|>\n"
            images.append(Image.open(image_path).convert("RGB"))
        question = (
            f"{user_prompt}{image_tokens}{prompt}{prompt_suffix}{assistant_prompt}"
        )

        inputs = self.processor(text=question, images=images, return_tensors="pt")
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=512,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        # Hack: remove the prefix question
        generated_text = generated_text.split(prompt)[1]
        return generated_text, None
