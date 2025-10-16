from openai import OpenAI
from .base_solver import Solver, encode_image
from utils import CostMeter
import time
import logging
import yaml

class GPT5Solver(Solver):
    def __init__(self, image_root, debug_mode, **gpt5_config):
        super().__init__(image_root, debug_mode)
        self.solver_name = gpt5_config.get("name", "GPT-5")
        self.gpt_config = gpt5_config
        with open(gpt5_config.get("api_yaml"), 'r') as api:
            key = yaml.safe_load(api)[gpt5_config.get("api_key")]
        self.client = OpenAI(
            api_key=key
        )

    def preprocessing(self):
        self.openai_usage = CostMeter(self.gpt_config.get("model", "gpt-5"))

    def generate(self, prompt, image_paths):
        content = [{"type": "input_text", "text": prompt}]

        for _, image_path in enumerate(image_paths):
            base64_image = encode_image(image_path)
            if self.gpt_config.get("low_res_mode", False):
                content += [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low",
                    }
                ]
            else:
                content += [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
        pred_ans = ""
        reasoning = ""
        response = ""
        metadata = ""
        for _ in range(5):
            try:
                response = self.client.responses.create(
                    model=self.gpt_config.get("model", "gpt-5"),
                    input=[{"role": "user", "content": content}],
                    reasoning={
                        "effort": "high",
                        "summary": "detailed"
                    }
                )
                self.openai_usage.update_responses(response.model_dump()["usage"])
                pred_ans = response.output_text
                reasoning = response.model_dump()["output"][0]["summary"]
                metadata = response.model_dump()
                break
            except Exception as e:
                logging.error(f"Error occured when calling LLM API: {e}")
                time.sleep(60)

        return pred_ans, None, reasoning, metadata

    def postprocessing(self):
        logging.info(f"Total cost: ${self.openai_usage.cost:.2f} 💸💸💸")