import anthropic
from .base_solver import Solver, encode_image
from utils import CostMeter
from types import SimpleNamespace
import time
import logging


class ClaudeSolver(Solver):
    def __init__(self, image_root, debug_mode, **config):
        super().__init__(image_root, debug_mode)
        self.solver_name = config.get("name", "Claude-3")
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.get("api_key"))

    def preprocessing(self):
        self.claude3_usage = CostMeter(
            self.config.get("model", "claude-3-haiku-20240307")
        )

    def generate(self, prompt, image_paths):
        content = []
        for _, image_path in enumerate(image_paths):
            base64_image = encode_image(image_path)
            content += [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "data": base64_image,
                        "media_type": "image/jpeg",
                    },
                }
            ]
        content += [{"type": "text", "text": prompt}]

        pred_ans = ""
        for _ in range(5):
            try:
                message = self.client.messages.create(
                    model=self.config.get("model", "gpt-4o"),
                    max_tokens=4096,
                    messages=[
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                )
                usage = SimpleNamespace(
                    prompt_tokens=message.usage.input_tokens,
                    completion_tokens=message.usage.output_tokens,
                )
                self.claude3_usage.update(usage)
                pred_ans = message.content[-1].text
                break
            except Exception as e:
                print(f"Error occured when calling LLM API: {e}")
                time.sleep(60)

        return pred_ans, None

    def postprocessing(self):
        logging.info(f"Total cost: ${self.claude3_usage.cost:.2f} 💸💸💸")
