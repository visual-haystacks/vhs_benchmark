from .base_solver import Solver
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch


class QwenVLSolver(Solver):
    def __init__(self, image_root, debug_mode, **config):
        super().__init__(image_root, debug_mode)
        self.solver_name = config.get("name", "Qwen/Qwen2-VL-7B-Instruct")
        self.config = config
        huggingface_model_id = config.get(
            "huggingface_model_id", "Qwen/Qwen2-VL-7B-Instruct"
        )
        self.processor = AutoProcessor.from_pretrained(huggingface_model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            huggingface_model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()

    @torch.inference_mode()
    def generate(self, prompt, image_paths):
        messages = [{"role": "user", "content": []}]
        for image_path in image_paths:
            messages[0]["content"].append({"type": "image", "image": image_path})

        messages[0]["content"].append({"type": "text", "text": prompt})
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            max_new_tokens=512,
            clean_up_tokenization_spaces=False,
        )[0]
        return response, None
