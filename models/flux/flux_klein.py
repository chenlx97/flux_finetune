import torch
from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor


def load_flux_components(pretrained_path, revision=None, variant=None, dtype=torch.bfloat16, device="cuda"):
    """加载 Flux2 Klein 相关组件"""
    vae = AutoencoderKLFlux2.from_pretrained(
        pretrained_path, subfolder="vae", revision=revision, variant=variant, torch_dtype=dtype
    )
    
    transformer = Flux2Transformer2DModel.from_pretrained(
        pretrained_path, subfolder="transformer", revision=revision, variant=variant, torch_dtype=dtype
    )
    
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        pretrained_path, subfolder="text_encoder", revision=revision, variant=variant
    )
    
    tokenizer = Qwen2TokenizerFast.from_pretrained(pretrained_path, subfolder="tokenizer", revision=revision)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_path, subfolder="scheduler", revision=revision)
    
    return {
        "vae": vae,
        "transformer": transformer,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
        "image_processor": Flux2ImageProcessor()
    }

def get_latent_stats(vae, device):
    """获取 VAE 的归一化统计量"""
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(device)
    return latents_bn_mean, latents_bn_std