from .base_solver import Solver
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from PIL import Image


class LLamaSolver(Solver):
    def __init__(self, image_root, debug_mode, **config):
        super().__init__(image_root, debug_mode)
        self.solver_name = config.get("name", "LLama3")
        self.config = config
        huggingface_model_id = config.get(
            "huggingface_model_id", "meta-llama/Meta-Llama-3-8B-Instruct"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(huggingface_model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            huggingface_model_id,
            torch_dtype=torch.bfloat16,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate(self, prompt, image_paths):
        messages = [
            {
                "role": "system",
                "content": "You are good at answering questions about images.",
            },
            {"role": "user", "content": prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            response = outputs[0][input_ids.shape[-1] :]

        return self.tokenizer.decode(response, skip_special_tokens=True), None
