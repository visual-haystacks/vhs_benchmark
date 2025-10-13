from openai import OpenAI
from .base_solver import Solver, encode_image
from utils import CostMeter
import time
import logging


class OpenrouterSolver(Solver):
    def __init__(self, image_root, debug_mode, **orouter_config):
        super().__init__(image_root, debug_mode)
        self.solver_name = orouter_config.get("name", "Openrouter_Solver")
        self.orouter_config = orouter_config

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=orouter_config.get("api_key"),
            default_headers={
                "HTTP-Referer": orouter_config.get("referer", ""),
                "X-Title": orouter_config.get("title", "Eval Harness"),
            },
        )

    def preprocessing(self):
        # use the model-id (snapshot) name for cost tracking
        model_id = self.orouter_config.get("model")
        self.openai_usage = CostMeter(model_id)

    def generate(self, prompt, image_paths):
        # build the content in the same way (multimodal) or just text-only
        content = [{"type": "text", "text": prompt}]
        for _, image_path in enumerate(image_paths):
            b64 = encode_image(image_path)
            if self.orouter_config.get("low_res_mode", False):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
                })
            else:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })

        pred = ""
        metadata = ""
        for _ in range(5):
            try:
                response = self.client.chat.completions.create(
                    model = self.orouter_config.get("model"),
                    messages=[{"role": "user", "content": content}],
                    
                )
                # If usage is present, update cost
                if hasattr(response, "usage"):
                    self.openai_usage.update(response.usage)

                pred = response.choices[0].message.content
                metadata = response.model_dump()
                
                break
            except Exception as e:
                logging.error(f"Error calling {self.orouter_config.get('model')} via OpenRouter: {e}")
                time.sleep(60)
        return pred, None, None, metadata

    def postprocessing(self):
        logging.info(f"Total cost: ${self.openai_usage.cost:.2f}")
