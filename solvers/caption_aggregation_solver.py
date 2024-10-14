from .base_solver import Solver
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch


class CaptionAggregationSolver(Solver):
    def __init__(self, image_root, debug_mode, **config):
        super().__init__(image_root, debug_mode)
        self.solver_name = config.get("name", "Llama3_LLava-v1.5-7b")
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # initialize language model

        lang_huggingface_model_id = config.get(
            "language_huggingface_model_id", "meta-llama/Meta-Llama-3.1-8B-Instruct"
        )
        self.language_tokenizer = AutoTokenizer.from_pretrained(
            lang_huggingface_model_id
        )
        self.language_model = AutoModelForCausalLM.from_pretrained(
            lang_huggingface_model_id,
            torch_dtype=torch.bfloat16,
        ).eval()
        self.language_model.to(self.device)

    def do_actual_qa(self, prompt, image_captions):
        prompt = prompt.replace(
            "You are given a set of images. Please answer the following question in Yes or No: ",
            "",
        )
        # Aggregate the image captions to the final prompt
        final_prompt = "Here are the captions for a set of images:"
        for i, caption in enumerate(image_captions, start=1):
            # caption = caption.replace("\n", " ")
            final_prompt += f"\n# Caption ({i})\n{caption}"
        final_prompt += f"\n\nBased on these image captions, please answer the following question: {prompt}. Please assume there must be at least one image satisfies the condition. Answer with 'Yes' or 'No' only."
        messages = [
            {
                "role": "system",
                "content": "You are a top expert in interpreting image captions and providing precise answers to questions based on the information they contain.",
            },
            {"role": "user", "content": final_prompt},
        ]

        input_ids = self.language_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        terminators = [
            self.language_tokenizer.eos_token_id,
            self.language_tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        with torch.inference_mode():
            outputs = self.language_model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                pad_token_id=self.language_tokenizer.eos_token_id,
            )
            response = outputs[0][input_ids.shape[-1] :]

        return self.language_tokenizer.decode(response, skip_special_tokens=True), None

    def generate(self, prompt, data_paths):
        image_captions = []
        assert (
            "captions" in data_paths[0]
        ), "LLamaLLaVASolver requires pre-generated captions"

        # assumes the data path contains pre-generated captions
        for caption_path in data_paths:
            caption_path = caption_path.replace(".jpg", ".txt")
            with open(caption_path, "r") as f:
                image_captions.append(f.read())
        # LLM aggregation
        return self.do_actual_qa(prompt, image_captions)
