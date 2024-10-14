from .base_solver import Solver
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN
import torch


class InternVLSolver(Solver):
    def __init__(self, image_root, debug_mode, **config):
        super().__init__(image_root, debug_mode)
        self.solver_name = config.get("name", "InternVL2-8B")
        self.config = config
        huggingface_model_id = config.get(
            "huggingface_model_id", "OpenGVLab/InternVL2-8B"
        )
        self.pipeline = pipeline(
            huggingface_model_id,
            backend_config=TurbomindEngineConfig(session_len=256000),
        )
        assert torch.cuda.is_available(), "lmdeploy requires CUDA"
        self.device = "cuda"

    def generate(self, prompt, image_paths):
        # Form Text and image prompts
        real_prompt = ""
        images = []
        for idx, image_path in enumerate(image_paths):
            images.append(load_image(image_path))
            real_prompt += f"Image-{idx+1}: {IMAGE_TOKEN}\n"
        real_prompt += prompt

        response = self.pipeline((real_prompt, images))
        return response.text, None
