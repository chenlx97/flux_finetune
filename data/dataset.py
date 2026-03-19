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

class DreamBoothDataset(Dataset):
    def __init__(
        self,
        instance_data_root: str,
        instance_prompt: str,
        buckets: List[Tuple[int, int]],
        repeats: int = 1,
        center_crop: bool = False,
        random_flip: bool = False,
        max_area: int = 1024 * 1024,
        multiple_of: int = 8,
    ):
        self.instance_prompt = instance_prompt
        self.buckets = buckets
        self.repeats = repeats
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.max_area = max_area
        self.multiple_of = multiple_of

        root = Path(instance_data_root)
        distorted_dir = root / "distorted"
        target_dir = root / "target"

        self.cond_paths = sorted(list(distorted_dir.iterdir()))
        self.instance_paths = sorted(list(target_dir.iterdir()))

        if len(self.instance_paths) != len(self.cond_paths):
            raise ValueError("distorted 和 target 数量不一致")

        self.num_images = len(self.instance_paths)
        self._length = self.num_images * repeats

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5], [0.5])

        self.bucket_ids = []
        for path in tqdm(self.instance_paths, desc="Processing images"):
            with Image.open(path) as img:
                img = exif_transpose(img.convert("RGB"))
                img = self.resize_if_needed(img)
            w, h = img.size
            bucket_idx = self.find_nearest_bucket(h, w)
            self.bucket_ids.append(bucket_idx)
        self.bucket_ids = self.bucket_ids * self.repeats

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        real_index = index % self.num_images
        instance_path = self.instance_paths[real_index]
        cond_path = self.cond_paths[real_index]

        with Image.open(instance_path) as img:
            image = exif_transpose(img.convert("RGB"))
        with Image.open(cond_path) as img:
            cond_image = exif_transpose(img.convert("RGB"))

        image = self.resize_if_needed(image)
        cond_image = self.resize_if_needed(cond_image)

        width, height = image.size
        bucket_idx = self.bucket_ids[index]
        target_h, target_w = self.buckets[bucket_idx]

        image = image.resize((target_w, target_h), Image.BILINEAR)
        cond_image = cond_image.resize((target_w, target_h), Image.BILINEAR)

        image = self.align_to_multiple(image)
        cond_image = self.align_to_multiple(cond_image)

        image, cond_image = self.paired_transform(image, cond_image)

        return {
            "instance_images": image,
            "cond_images": cond_image,
            "bucket_idx": bucket_idx,
            "instance_prompt": self.instance_prompt,
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

    def find_nearest_bucket(self, height, width):
        ratios = [abs((h / w) - (height / width)) for h, w in self.buckets]
        return ratios.index(min(ratios))

    def paired_transform(self, image, cond_image):
        if not self.center_crop:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=image.size[::-1])
            image = TF.crop(image, i, j, h, w)
            cond_image = TF.crop(cond_image, i, j, h, w)

        if self.random_flip and random.random() < 0.5:
            image = TF.hflip(image)
            cond_image = TF.hflip(cond_image)

        image = self.normalize(self.to_tensor(image))
        cond_image = self.normalize(self.to_tensor(cond_image))
        return image, cond_image

class BucketBatchSampler(BatchSampler):
    def __init__(self, dataset: DreamBoothDataset, batch_size: int, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        bucket_indices = [[] for _ in range(len(self.dataset.buckets))]
        for idx, bucket_idx in enumerate(self.dataset.bucket_ids):
            bucket_indices[bucket_idx].append(idx)
        all_batches = []
        for indices in bucket_indices:
            random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)
        random.shuffle(all_batches)
        return iter(all_batches)

    def __len__(self):
        total = len(self.dataset.bucket_ids)
        if self.drop_last:
            return total // self.batch_size
        else:
            return (total + self.batch_size - 1) // self.batch_size

def collate_fn(examples):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts}
    if any("cond_images" in example for example in examples):
        cond_pixel_values = [example["cond_images"] for example in examples]
        cond_pixel_values = torch.stack(cond_pixel_values)
        cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
        batch.update({"cond_pixel_values": cond_pixel_values})
    return batch