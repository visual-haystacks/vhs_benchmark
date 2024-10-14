from .base_solver import Solver
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image

# LLaVA's packages
# Please install them before running the code
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


class ViLASolver(Solver):
    def __init__(self, image_root, debug_mode, **config):
        super().__init__(image_root, debug_mode)
        self.solver_name = config.get("name", "LLaVA-v1.5-7b-Official")
        self.config = config
        huggingface_model_id = config.get(
            "huggingface_model_id", "liuhaotian/llava-v1.5-7b"
        )
        model_name = config.get("model_name", "llava-v1.5-7b")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = get_model_name_from_path(huggingface_model_id)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            huggingface_model_id,
            model_name,
            None,
            False,
            False,
            device=self.device,
            device_map="auto",
        )

        self.processor = image_processor
        self.tokenizer = tokenizer
        self.model = model.eval()

    def generate(self, prompt, image_paths):
        # initialize converstaions
        conv = conv_templates["llama_3"].copy()
        # prepare images
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        image_size = [image.size for image in images]
        image_tensor = process_images(images, self.processor, self.model.config)
        image_tensor = image_tensor.to(self.device, dtype=torch.float16)
        # prepare texts
        inp = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + prompt

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.inference_mode():
            generated_id = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=512,
                use_cache=True,
            )
        generated_text = (
            self.tokenizer.decode(generated_id[0])
            .replace("<|begin_of_text|>", "")
            .replace("<|end_of_text|>", "")
            .strip()
        )

        return generated_text, None
