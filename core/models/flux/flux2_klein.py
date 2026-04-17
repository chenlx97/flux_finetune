import torch
from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from registry.model_registry import ModelRegistry


@ModelRegistry.register('flux2_klein')
class Flux2KleinModel:
    def __init__(self, config, dtype=torch.bfloat16, device="cuda"):

        self.config = config
        self.device = device
        self.dtype = dtype
        self.pipe = Flux2KleinPipeline.from_pretrained(
            self.config.model.pretrained_model_name_or_path,
            torch_dtype=dtype
        ).to("cpu")
        self.vae = self.pipe.vae
        self.transformer = self.pipe.transformer
        self.text_encoder = self.pipe.text_encoder
        self.scheduler = self.pipe.scheduler
        self.image_processor = Flux2ImageProcessor()
        self._compute_latent_stats()

    def _compute_latent_stats(self):
        if hasattr(self.vae, 'bn'):
            self.latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(self.device)
            self.latents_bn_std = torch.sqrt(
                self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
            ).to(self.device)
        else:
            self.latents_bn_mean = torch.zeros(1, 1, 1, 1).to(self.device)
            self.latents_bn_std = torch.ones(1, 1, 1, 1).to(self.device)

    def to(self, device):
        self.device = device
        self.vae.to(device)
        self.transformer.to(device)
        return self

    def set_trainable(self, trainable=True):
        self.transformer.requires_grad_(trainable)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
    @torch.no_grad()
    def _encode(self, prompt):
        if next(self.text_encoder.parameters()).device.type != self.device:
            self.text_encoder = self.text_encoder.to(self.device).eval()
        prompt_embeds, text_ids = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            max_sequence_length=self.config.model.max_sequence_length,
        )
        return prompt_embeds, text_ids

    def unload_text_encoder(self):
        if self.text_encoder is not None:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()