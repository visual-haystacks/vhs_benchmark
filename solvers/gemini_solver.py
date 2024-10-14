from .base_solver import Solver
import google.generativeai as genai
from types import SimpleNamespace
from PIL import Image
from utils import CostMeter
import logging
import time


# We follow the official example jupyter notebook to put image after text
# URL: https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/gemini-api/docs/get-started/python.ipynb
class GeminiSolver(Solver):
    def __init__(self, image_root, debug_mode, **gemini_config):
        super().__init__(image_root, debug_mode)
        self.solver_name = gemini_config.get("name", "Gemini-1.5-pro")
        self.gemini_config = gemini_config
        genai.configure(api_key=gemini_config.get("api_key"))
        """
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
        """
        self.model = genai.GenerativeModel(
            gemini_config.get("model", "gemini-1.5-pro-001")
        )

    def preprocessing(self):
        self.gemini_usage = CostMeter(
            self.gemini_config.get("model", "gemini-1.5-pro-001")
        )

    def generate(self, prompt, image_paths):
        content = []
        # enable this line if we're running gemini-1.0
        # image_paths = image_paths[:16]
        # f = genai.upload_file(path); m.generate_content(['tell me about this file:', f])
        if "gemini-1.0" in self.gemini_config.get("model", "gemini-1.5-pro-001"):
            image_paths = image_paths[:16]
        for idx, image_path in enumerate(image_paths):
            if len(image_paths) <= 100:
                content.append(Image.open(image_path))
            else:
                f = genai.upload_file(image_path)
                content.append(f)
        content.append(prompt)
        # Hack to avoid the issue with the image being blocked
        # safety_settings = [
        #     {
        #         "category": "HARM_CATEGORY_DANGEROUS",
        #         "threshold": "BLOCK_NONE",
        #     },
        #     {
        #         "category": "HARM_CATEGORY_HARASSMENT",
        #         "threshold": "BLOCK_NONE",
        #     },
        #     {
        #         "category": "HARM_CATEGORY_HATE_SPEECH",
        #         "threshold": "BLOCK_NONE",
        #     },
        #     {
        #         "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        #         "threshold": "BLOCK_NONE",
        #     },
        #     {
        #         "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        #         "threshold": "BLOCK_NONE",
        #     },
        # ]

        pred_ans = ""

        for _ in range(5):
            try:
                # response = self.model.count_tokens(content)
                # print(f"Prompt Character Count: {response.total_billable_characters}")
                response = self.model.generate_content(content)

                # usage = SimpleNamespace(
                #     prompt_tokens=response.usage_metadata.prompt_token_count,
                #     completion_tokens=response.usage_metadata.candidates_token_count,
                # )
                # self.gemini_usage.update(usage)
                if len(response.candidates) > 0:
                    pred_ans = response.candidates[0].content.parts[0].text
                else:
                    print(f"Rejected by the model: {response.prompt_feedback}")
                break
            except Exception as e:
                print(f"Error occured when calling LLM API: {e}")
                time.sleep(60)

        return pred_ans, None

    def postprocessing(self):
        logging.info(f"Total cost: ${self.gemini_usage.cost:.2f} 💸💸💸")
