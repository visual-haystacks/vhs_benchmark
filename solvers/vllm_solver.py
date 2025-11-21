import json
import random
import os
from vllm import LLM, SamplingParams
from .base_solver import Solver
from PIL import Image


class VLLMSolver(Solver):
    def __init__(self, image_root, debug_mode, **config):
        super().__init__(image_root, debug_mode)
        self.solver_name = config["name"]
        self.config = config
        self.huggingface_model_id = config["huggingface_model_id"]
        self.model = LLM(model=self.huggingface_model_id, 
            tensor_parallel_size=config.get("tensor_parallel_size", 1),
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            disable_mm_preprocessor_cache=True
        )
        self.sampling_params = SamplingParams(
            max_tokens=config.get("max_new_tokens", self.max_new_tokens),
            temperature=config.get("temperature", self.temperature),
        )
        self.batch_size = config.get("batch_size", 10)

    def prepare_inputs_for_vllm(self, prompt, image_lists):
        v_placeholder = [{"type": "image_pil", "image_pil": Image.open(os.path.join(self.image_root, image)).convert("RGB")} for image in image_lists]
        return [
            {"role": "user",
                "content": [
                    *v_placeholder,
                    {"type": "text", "text": prompt},
                ]
            }
        ]

    def run_fast(self, test_file, output_dir):
        self.preprocessing()
        test_data = json.load(open(test_file, "r"))
        if self.debug_mode:
            test_data = test_data[::3]
        # Step 1: Create batchs of prompts and images
        inputs = []
        for entry in test_data:
            prompt = (
                "You are given a set of images. Please answer the following question in Yes or No: "
                + entry["conversations"][0]["value"]
            )
            image_lists = entry["pos_image"] + entry["neg_image"]
            random.shuffle(image_lists)
            inputs.append(self.prepare_inputs_for_vllm(prompt, image_lists))
        
        # Do batch generation to prevent memory issue in preprocessing
        out_texts = [None] * len(test_data)
        for start in range(0, len(inputs), self.batch_size):
            chunk = inputs[start:start+self.batch_size]
            outs = self.model.chat(chunk, sampling_params=self.sampling_params)
            for idx, output in enumerate(outs):
                out_texts[start+idx] = output.outputs[0].text

        for idx, generated_text in enumerate(out_texts):
            output_fname = os.path.join(output_dir, f"{idx:03d}.json")
            # Temporarily solution for Qwen3-VL-8B-Thinking
            if "</think>" in generated_text:
                reasoning, generated_text = generated_text.split("</think>")
                reasoning = reasoning.strip()
                generated_text = generated_text.strip()
            else:
                reasoning = ""
            entry = test_data[idx]
            entry["result"] = {
                "image_paths": inputs[idx]['image_paths'],
                "response": generated_text,
                "log": "",
                "reasoning": reasoning,
                "all_metadata": "",
            }
            with open(output_fname, "w") as f:
                json.dump(entry, f, indent=2)