import gradio as gr
import numpy as np
import random
import torch
import spaces
from diffusers import AutoModel, DiffusionPipeline, TorchAoConfig # type: ignore
from diffusers import Flux2Pipeline
from PIL import Image
import os
from tqdm import tqdm
from torchao.dtypes.affine_quantized_tensor import AffineQuantizedTensor
def _safe_has_compatible_shallow_copy_type(t1, t2):
    return True
torch._has_compatible_shallow_copy_type = _safe_has_compatible_shallow_copy_type
AffineQuantizedTensor.__torch_function__ = torch._C._disabled_torch_function_impl



# ========= 1. 创建 pipeline =========
def create_pipe():
    model_dir = "/data/clx/control-lora-v2-master/ckpt/FLUX.2-dev-bnb-4bit"
    pipe = Flux2Pipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    return pipe


# ========= 2. prompt 组件 =========
ROOMS = [
    "bedroom", "living room", "kitchen", "dining room"
]

VIEWS = [
    "wide angle", "corner view"
]

LIGHTS = [
    "soft natural light", "morning light", "warm lighting", "daylight"
]

STYLES = [
    "nordic style"
]

DETAILS = [
    "wooden furniture", "neutral color palette", "cozy atmosphere",
    "clean layout", "modern furniture", "natural materials"
]


def build_prompt():
    room = random.choice(ROOMS)
    view = random.choice(VIEWS)
    light = random.choice(LIGHTS)
    style = random.choice(STYLES)
    detail1 = random.choice(DETAILS)
    detail2 = random.choice(DETAILS)

    prompt = f"{style} {room}, {detail1}, {detail2}, {light}, {view}"
    return prompt

def generate_dataset(pipe, save_dir="./dataset", num_images=50):
    os.makedirs(save_dir, exist_ok=True)

    for i in tqdm(range(num_images)):
        prompt = build_prompt()

        image = pipe(
            prompt=prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            height=1024,
            width=1024,
        ).images[0]

        # ===== 保存图片 =====
        img_path = os.path.join(save_dir, f"{i:03d}.png")
        image.save(img_path)

        # ===== 保存caption =====
        txt_path = os.path.join(save_dir, f"{i:03d}.txt")

        # 加 trigger word
        caption = f"<nordic_style> {prompt}"

        with open(txt_path, "w") as f:
            f.write(caption)


# ========= 5. 运行 =========
if __name__ == "__main__":
    pipe = create_pipe()
    generate_dataset(pipe, save_dir="./nordic_dataset", num_images=50)