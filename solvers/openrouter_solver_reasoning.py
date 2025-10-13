from openai import OpenAI
from .base_solver import Solver, encode_image
from utils import CostMeter
import time
import logging
import requests
import json


class OpenrouterSolver(Solver):
    def __init__(self, image_root, debug_mode, **orouter_config):
        super().__init__(image_root, debug_mode)
        self.solver_name = orouter_config.get("name")
        self.orouter_config = orouter_config
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers={
            'Authorization': f"Bearer {orouter_config.get('api_key')}",
            'Content-Type': 'application/json',
        }

    def preprocessing(self):
        self.openai_usage = CostMeter(self.orouter_config.get("model"))

    def generate(self, prompt, image_paths):
        content = [{"type": "text", "text": prompt}]

        for _, image_path in enumerate(image_paths):
            base64_image = encode_image(image_path)
            if self.orouter_config.get("low_res_mode", False):
                content += [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":f"data:image/jpeg;base64,{base64_image}",
                        },
                        "detail": "low",
                    }
                ]
            else:
                content += [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":f"data:image/jpeg;base64,{base64_image}",
                        }
                    }
                ]

        m = [
                {
                    "role": "user",
                    "content": content
            }
        ]
        pred_ans = ""
        reasoning = ""
        response = ""
        metadata = ""
        for _ in range(5):
            try:
                # use responses api - provides reasoning trace

                payload = {
                    "model" : self.orouter_config.get("model"),
                    "messages": m,
                    "reasoning": {
                        "max_tokens": 2000 #, can use either "effort" or "max_tokens" for models like Claude, Grok - model specific setting. See: https://openrouter.ai/docs/use-cases/reasoning-tokens 
                    }
                }
                response = requests.post(
                    self.url,
                    headers = self.headers,
                    data=json.dumps(payload)
                )
            
                reasoning = response.json()['choices'][0]['message']['reasoning']
                pred_ans = response.json()['choices'][0]['message']['content']
                metadata = response.json()

                
                
                break
            except Exception as e:
                logging.error(f"Error occured when calling LLM API: {e}")
                time.sleep(60)

        return pred_ans, None, reasoning, metadata

    def postprocessing(self):
        logging.info(f"Total cost: ${self.openai_usage.cost:.2f} 💸💸💸")