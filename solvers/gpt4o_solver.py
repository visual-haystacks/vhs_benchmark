from openai import OpenAI
from .base_solver import Solver, encode_image
from utils import CostMeter
import time
import logging


class GPT4OSolver(Solver):
    def __init__(self, image_root, debug_mode, **gpt4_config):
        super().__init__(image_root, debug_mode)
        self.solver_name = gpt4_config.get("name", "GPT-4o")
        self.gpt_config = gpt4_config
        self.client = OpenAI(
            api_key=gpt4_config.get("api_key"),
            organization=gpt4_config.get("organization", None),
        )

    def preprocessing(self):
        self.openai_usage = CostMeter(self.gpt_config.get("model", "gpt-4o"))

    def generate(self, prompt, image_paths):
        content = [{"type": "text", "text": prompt}]

        for _, image_path in enumerate(image_paths):
            base64_image = encode_image(image_path)
            if self.gpt_config.get("low_res_mode", False):
                content += [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low",
                        },
                    }
                ]
            else:
                content += [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                ]
        pred_ans = ""
        for _ in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.gpt_config.get("model", "gpt-4o"),
                    messages=[{"role": "user", "content": content}],
                )
                self.openai_usage.update(response.usage)
                pred_ans = response.choices[0].message.content
                break
            except Exception as e:
                logging.error(f"Error occured when calling LLM API: {e}")
                time.sleep(60)

        return pred_ans, None

    def postprocessing(self):
        logging.info(f"Total cost: ${self.openai_usage.cost:.2f} 💸💸💸")
