import json
import random
import os
from vllm import LLM, SamplingParams
# from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoProcessor
from .base_solver import Solver


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
        )
        self.processor = AutoProcessor.from_pretrained(self.huggingface_model_id)
        self.sampling_params = SamplingParams(
            max_tokens=config.get("max_new_tokens", self.max_new_tokens),
            temperature=config.get("temperature", self.temperature),
        )
        self.batch_size = config.get("batch_size", 20)

    def prepare_inputs_for_vllm(self, prompt, image_lists):
        conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for image_path in image_lists:
            conversation[0]["content"].append({"type": "image", "image": image_path})
        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        mm_data = {}
        mm_data['image'] = [os.path.join(self.image_root, img) for img in image_lists]
        return {
            'prompt': text,
            'multi_modal_data': mm_data,
        }

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
            mm_data = self.prepare_inputs_for_vllm(prompt, image_lists)
            inputs.append(mm_data)
        
        # Do batch generation to prevent memory issue in preprocessing
        out_texts = [None] * len(test_data)
        for start in range(0, len(inputs), self.batch_size):
            chunk = inputs[start:start+self.batch_size]
            outs = self.model.generate(chunk, sampling_params=self.sampling_params)
            for idx, output in enumerate(outs):
                out_texts[start+idx] = output.outputs[0].text

        for idx, generated_text in enumerate(out_texts):
            output_fname = os.path.join(output_dir, f"{idx:03d}.json")
            entry = test_data[idx]
            entry["result"] = {
                "image_paths": inputs[idx]['multi_modal_data']['image'],
                "response": generated_text,
                "log": "",
                "reasoning": "",
                "all_metadata": "",
            }
            with open(output_fname, "w") as f:
                json.dump(entry, f, indent=2)