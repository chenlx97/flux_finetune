import os
import torch
import hashlib
from tqdm import tqdm
from utils.make_datajson import load_dataset
from diffusers.training_utils import offload_models

def hash_prompt(prompt: str):
    return hashlib.md5(prompt.encode()).hexdigest()


class TextPrecompute:
    def __init__(
        self,
        model_wrapper,
        config,
    ):
        
        self.model_wrapper = model_wrapper
        self.config = config
        self.max_sequence_length = self.config.model.max_sequence_length
        self.save_dir = self.config.training.output_dir+"/textcache"
        os.makedirs(self.save_dir, exist_ok=True)

    def _get_path(self, prompt):
        key = hash_prompt(prompt)
        return os.path.join(self.save_dir, f"{key}.pt")

    def run(self):
        data_list = load_dataset(self.config.data.data_json)
        for data in tqdm(data_list):
            prompt = data["prompt"]
            path = self._get_path(prompt)
            if os.path.exists(path):
                continue
            prompt_embeds, text_ids = self.model_wrapper._encode(prompt)
            torch.save(
                {
                    "prompt_embeds": prompt_embeds.cpu(),
                    "text_ids": text_ids.cpu(),
                },
                path,
            )
        self.model_wrapper.unload_text_encoder()

    def exists(self, prompt):
        return os.path.exists(self._get_path(prompt))

    def load(self, prompt, map_location="cpu"):
        path = self._get_path(prompt)
        return torch.load(path, map_location=map_location)