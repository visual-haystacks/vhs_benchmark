from tqdm import tqdm
import random
import base64
import json
import os


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class Solver:
    def __init__(self, image_root, debug_mode):
        self.solver_name = "Base Solver"
        self.image_root = image_root
        self.debug_mode = debug_mode

    def preprocessing(self):
        # Any preprocessing steps can be added here
        pass

    def postprocessing(self):
        # Any postprocessing steps can be added here
        pass

    def generate(self, prompt, image_paths, **kwargs):
        # This should be implemented with the actual generation logic
        raise NotImplementedError

    def construct_image_paths(self, entry):
        # Handle single and multi needle scenarios
        if len(entry["pos_image"]) == 1:
            # Single Needle Scenario
            pos_img_path = entry["pos_image"][0]
            neg_img_paths = entry["neg_image"]
            total_images = len(neg_img_paths)
            needle_pos = random.randint(0, total_images)
            image_paths = (
                neg_img_paths[:needle_pos] + [pos_img_path] + neg_img_paths[needle_pos:]
            )
        else:
            # Multi Needle Scenario
            pos_img_paths = entry["pos_image"]
            neg_img_paths = entry["neg_image"]
            image_paths = pos_img_paths + neg_img_paths
            random.shuffle(image_paths)
        return image_paths

    def run_detailed(self, test_file, output_dir):
        # Only support single mode, ablating needle position
        self.preprocessing()
        test_data = json.load(open(test_file, "r"))
        if self.debug_mode:
            test_data = test_data[:1]

        for idx, entry in tqdm(enumerate(test_data), total=len(test_data)):
            # Single Needle: For the image with [xxx], is there a [yyy]?
            prompt = entry["conversations"][0]["value"] + " Answer Yes or No."
            # shuffle input iamge orderings
            num_images = len(entry["pos_image"]) + len(entry["neg_image"])
            interval = max(int(num_images // 10), 1)
            for needle_pos in range(0, num_images, interval):
                image_lists = (
                    entry["neg_image"][:needle_pos]
                    + entry["pos_image"]
                    + entry["neg_image"][needle_pos:]
                )
                # use the right image root directory
                image_paths = [
                    os.path.join(self.image_root, img) for img in image_lists
                ]
                response, log_msg = self.generate(
                    prompt, image_paths
                )  # Adjusted to use self.generate
                # Store the result in the entry
                if "response" not in entry:
                    entry["response"] = []
                entry["response"].append(response)
            with open(os.path.join(output_dir, f"{idx:03d}.json"), "w") as f:
                json.dump(entry, f)
        self.postprocessing()

    def run_fast(self, test_file, output_dir):
        self.preprocessing()
        test_data = json.load(open(test_file, "r"))
        if self.debug_mode:
            test_data = test_data[:1]

        for idx, entry in tqdm(enumerate(test_data), total=len(test_data)):
            # Single Needle: For the image with [xxx], is there a [yyy]?
            prompt = entry["conversations"][0]["value"] + " Answer Yes or No."
            # shuffle input iamge orderings
            image_lists = entry["pos_image"] + entry["neg_image"]
            random.shuffle(image_lists)

            # use the right image root directory
            image_paths = [os.path.join(self.image_root, img) for img in image_lists]
            response, log_msg = self.generate(
                prompt, image_paths
            )  # Adjusted to use self.generate
            # Store the result in the entry
            entry["result"] = {
                "image_paths": image_lists,
                "response": response,
                "log": log_msg,
            }
            with open(os.path.join(output_dir, f"{idx:03d}.json"), "w") as f:
                json.dump(entry, f)
        self.postprocessing()
