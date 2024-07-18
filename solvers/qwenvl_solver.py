from .base_solver import Solver
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# This file is still broken now ;(


class QwenVLSolver(Solver):
    def __init__(self, image_root, debug_mode, **config):
        super().__init__(image_root, debug_mode)
        self.solver_name = config.get("name", "Qwen/Qwen-7B-Chat")
        self.config = config
        huggingface_model_id = config.get("huggingface_model_id", "Qwen/Qwen-7B-Chat")
        self.tokenizer = AutoTokenizer.from_pretrained(
            huggingface_model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            huggingface_model_id,
            trust_remote_code=True,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate(self, prompt, image_paths):
        # Qwen-VL (https://arxiv.org/abs/2308.12966) handles at most 8 images
        # 2048 context length / 256 tokens per image = 8 images
        image_paths = image_paths[:10]
        content = []
        for idx, image_path in enumerate(image_paths):
            content.append({"image": image_path})
        content.append({"text": prompt})
        query = self.tokenizer.from_list_format(content)

        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        return response, None
