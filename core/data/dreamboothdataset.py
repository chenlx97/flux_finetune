import os
import random
from pathlib import Path
from typing import List, Tuple
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset, BatchSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
from utils.make_datajson import load_dataset
from registry.dataset_registry import DatasetRegistry

@DatasetRegistry.register('dreamboothdataset')
class DreamBoothDataset(Dataset):
    
    def __init__(
        self,
        config,
        text_emb = None,
        max_area: int = 1024 * 1024,
        multiple_of: int = 8,
    ):

        self.config = config
        self.resolution = self.config.data.resolution
        self.repeats = self.config.data.repeats
        self.max_area = max_area
        self.multiple_of = multiple_of
        self.text_emb = text_emb
        self.data_list = load_dataset(self.config.data.data_json)
        self.num_images = len(self.data_list)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5], [0.5])
        self.bucket_ids = None

    def __len__(self):
        return len(self.data_list) * self.repeats

    def __getitem__(self, index):
        real_index = index % self.num_images
        target_path = self.data_list[real_index]['target']
        cond_path = self.data_list[real_index]['condition']
        prompt = self.data_list[real_index]['prompt']
        target_h, target_w = self.resolution
        with Image.open(target_path) as img:
            target_image = exif_transpose(img.convert("RGB"))
        target_image = self.resize_if_needed(target_image)
        target_image = target_image.resize((target_w, target_h), Image.BILINEAR)
        target_image = self.align_to_multiple(target_image)
        target_image = self.normalize(self.to_tensor(target_image))

        if cond_path:
            with Image.open(cond_path) as img:
                cond_image = exif_transpose(img.convert("RGB"))
            cond_image = self.resize_if_needed(cond_image)
            cond_image = cond_image.resize((target_w, target_h), Image.BILINEAR)
            cond_image = self.align_to_multiple(cond_image)
            cond_image = self.normalize(self.to_tensor(cond_image))
        else:
            cond_image = None

        text_embeding = self.text_emb.load(prompt)
        prompt_emb,text_ids = text_embeding["prompt_embeds"][0],text_embeding['text_ids'][0]
        return {
            "target_image": target_image,
            "cond_images": cond_image,
            "prompt_emb": prompt_emb,
            "text_ids": text_ids,
        }

    def resize_if_needed(self, img):
        w, h = img.size
        if w * h > self.max_area:
            scale = (self.max_area / (w * h)) ** 0.5
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)
        return img

    def align_to_multiple(self, img):
        w, h = img.size
        w = (w // self.multiple_of) * self.multiple_of
        h = (h // self.multiple_of) * self.multiple_of
        return img.resize((w, h), Image.BILINEAR)

    @staticmethod
    def collate_fn(examples):
        pixel_values = [example["target_image"] for example in examples]
        prompt_emb = [example["prompt_emb"] for example in examples]
        text_ids = [example["text_ids"] for example in examples]
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        prompt_emb = torch.stack(prompt_emb)
        prompt_emb = prompt_emb.to(memory_format=torch.contiguous_format).float()

        text_ids = torch.stack(text_ids)
        text_ids = text_ids.to(memory_format=torch.contiguous_format).int() 
        
        batch = {"pixel_values": pixel_values, "prompt_emb": prompt_emb,"text_ids": text_ids}
        if any(example["cond_images"] for example in examples):
            cond_pixel_values = [example["cond_images"] for example in examples]
            cond_pixel_values = torch.stack(cond_pixel_values)
            cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
            batch.update({"cond_pixel_values": cond_pixel_values})
        return batch