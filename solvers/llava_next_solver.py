import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from .base_solver import Solver


class LLaVANextSolver(Solver):
    def __init__(self, image_root, debug_mode, **config):
        super().__init__(image_root, debug_mode)
        self.solver_name = config.get("name", "LLaVA-Next")
        self.config = config
        self.huggingface_model_id = config.get(
            "huggingface_model_id", "llava-hf/llava-v1.6-mistral-7b-hf"
        )
        self.processor = LlavaNextProcessor.from_pretrained(
            self.huggingface_model_id, trust_remote_code=True
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.huggingface_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).eval()

    @torch.inference_mode()
    def generate(self, prompt, image_paths):

        images = []
        conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for img_num, image_path in enumerate(image_paths):
            conversation[0]["content"].append({"type": "image"})
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)

        formatted_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            text=formatted_prompt, images=images, return_tensors="pt"
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        print("Generated text: ", generated_text)
        # Hack: remove the prefix question
        generated_text = generated_text.split("[/INST]")[1]
        return generated_text, None
