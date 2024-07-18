from .base_solver import Solver
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
from PIL import Image


class LLamaLLaVASolver(Solver):
    def __init__(self, image_root, debug_mode, **config):
        super().__init__(image_root, debug_mode)
        self.solver_name = config.get("name", "Llama3_LLava-v1.5-7b")
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # initialize language model
        lang_huggingface_model_id = config.get(
            "language_huggingface_model_id", "meta-llama/Meta-Llama-3-8B-Instruct"
        )
        self.language_tokenizer = AutoTokenizer.from_pretrained(
            lang_huggingface_model_id
        )
        self.language_model = AutoModelForCausalLM.from_pretrained(
            lang_huggingface_model_id,
            torch_dtype=torch.bfloat16,
        )
        self.language_model.to(self.device)
        # initialize vision model
        vision_huggingface_model_id = config.get(
            "vision_huggingface_model_id", "llava-hf/llava-1.5-7b-hf"
        )
        self.vision_processor = AutoProcessor.from_pretrained(
            vision_huggingface_model_id
        )
        self.vision_model = LlavaForConditionalGeneration.from_pretrained(
            vision_huggingface_model_id,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vision_model.to(self.device)

    def generate_image_caption(self, image_path):
        instruction = "Please provide a detailed caption of the given image."
        question = f"USER: <image>\n{instruction}\nASSISTANT:"
        inputs = self.vision_processor(
            text=question, images=Image.open(image_path), return_tensors="pt"
        ).to(self.device)

        generated_id = self.vision_model.generate(**inputs, max_new_tokens=500)
        generated_text = self.vision_processor.batch_decode(
            generated_id, skip_special_tokens=True
        )[0]
        # Hack: remove the prefix question
        generated_text = generated_text.split("ASSISTANT: ")[1]
        return generated_text

    def do_actual_qa(self, prompt, image_captions):
        # Aggregate the image captions to the final prompt
        final_prompt = "Here are the captions for a set of images:"
        for i, caption in enumerate(image_captions, start=1):
            caption = caption.replace("\n", " ")
            final_prompt += f"\n{i}. {caption}"
        final_prompt += f"\n\nBased on these image captions, please answer the following question: {prompt}"

        messages = [
            {
                "role": "system",
                "content": "You are good at answering questions about images based on their captions.",
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
        # generate captions for each image
        if (
            "captions" in data_paths[0]
        ):  # assumes the data path contains pre-generated captions
            for caption_path in data_paths:
                caption_path = caption_path.replace(".jpg", ".txt")
                with open(caption_path, "r") as f:
                    image_captions.append(f.read())
        else:  # raw images
            for image_path in data_paths:
                image_captions.append(self.generate_image_caption(image_path))
        # LLM aggregation
        return self.do_actual_qa(prompt, image_captions)
