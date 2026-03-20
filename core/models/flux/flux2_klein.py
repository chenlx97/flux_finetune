import torch
from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from registry.model_registry import ModelRegistry

@ModelRegistry.register('flux2_klein')
class Flux2KleinModel:
    def __init__(self, pretrained_path, revision=None, variant=None, dtype=torch.bfloat16, device="cuda"):
        self.device = device
        self.dtype = dtype
        
        self.vae = AutoencoderKLFlux2.from_pretrained(
            pretrained_path, subfolder="vae", revision=revision, variant=variant, torch_dtype=dtype
        )
        
        self.transformer = Flux2Transformer2DModel.from_pretrained(
            pretrained_path, subfolder="transformer", revision=revision, variant=variant, torch_dtype=dtype
        )
        
        self.text_encoder = Qwen3ForCausalLM.from_pretrained(
            pretrained_path, subfolder="text_encoder", revision=revision, variant=variant
        )
        
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(pretrained_path, subfolder="tokenizer", revision=revision)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_path, subfolder="scheduler", revision=revision)
        self.image_processor = Flux2ImageProcessor()
        
        # 预计算 VAE 统计量
        self.latents_bn_mean = None
        self.latents_bn_std = None
        self._compute_latent_stats()

    def _compute_latent_stats(self):
        if hasattr(self.vae, 'bn'):
            self.latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(self.device)
            self.latents_bn_std = torch.sqrt(
                self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
            ).to(self.device)
        else:
            # 默认值，防止报错
            self.latents_bn_mean = torch.zeros(1, 1, 1, 1).to(self.device)
            self.latents_bn_std = torch.ones(1, 1, 1, 1).to(self.device)

    def to(self, device):
        self.vae.to(device)
        self.transformer.to(device)
        self.text_encoder.to(device)
        self.device = device
        self._compute_latent_stats()
        return self

    def set_trainable(self, trainable=True):
        self.transformer.requires_grad_(trainable)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)