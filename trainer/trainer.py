import os
import torch
from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.checkpoint import save_lora_checkpoint
from .loss import compute_flow_matching_loss

logger = get_logger(__name__)

class FluxTrainer:
    def __init__(self, accelerator: Accelerator, config):
        self.accelerator = accelerator
        self.config = config
        self.global_step = 0
        
    def train(self, train_dataloader, transformer, optimizer, lr_scheduler, noise_scheduler, 
              vae, text_encoding_pipeline, validation_fn=None):
        transformer.train()
        progress_bar = tqdm(range(self.config.training.max_train_steps), disable=not self.accelerator.is_local_main_process)
        
        latents_bn_mean, latents_bn_std = self._get_latent_stats(vae)
        
        for epoch in range(self.config.training.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate([transformer]):
                    loss = self._train_step(
                        batch, transformer, vae, text_encoding_pipeline, 
                        noise_scheduler, latents_bn_mean, latents_bn_std
                    )
                    
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(transformer.parameters(), self.config.training.max_grad_norm)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.global_step += 1
                    
                    if self.global_step % self.config.training.checkpointing_steps == 0:
                        self._save_checkpoint(transformer, epoch)
                    
                    if validation_fn and epoch % self.config.validation.validation_epochs == 0:
                        validation_fn(self.global_step)
                
                if self.global_step >= self.config.training.max_train_steps:
                    break
            
            if self.global_step >= self.config.training.max_train_steps:
                break
        
        progress_bar.close()
        self._save_final_checkpoint(transformer)

    def _train_step(self, batch, transformer, vae, text_encoding_pipeline, noise_scheduler, latents_bn_mean, latents_bn_std):
        # 简化版训练步骤，具体逻辑参考原脚本
        # 这里需要实现具体的 forward 和 loss 计算
        # 为保持简洁，此处调用 loss 模块
        device = self.accelerator.device
        pixel_values = batch["pixel_values"].to(device)
        cond_pixel_values = batch["cond_pixel_values"].to(device)
        
        # Encode latents
        model_input = vae.encode(pixel_values).latent_dist.mode()
        cond_model_input = vae.encode(cond_pixel_values).latent_dist.mode()
        
        # Patchify & Norm
        from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
        model_input = Flux2KleinPipeline._patchify_latents(model_input)
        model_input = (model_input - latents_bn_mean) / latents_bn_std
        
        cond_model_input = Flux2KleinPipeline._patchify_latents(cond_model_input)
        cond_model_input = (cond_model_input - latents_bn_mean) / latents_bn_std
        
        # Prepare IDs
        model_input_ids = Flux2KleinPipeline._prepare_latent_ids(model_input).to(device=device)
        cond_model_input_list = [cond_model_input[i].unsqueeze(0) for i in range(cond_model_input.shape[0])]
        cond_model_input_ids = Flux2KleinPipeline._prepare_image_ids(cond_model_input_list).to(device=cond_model_input.device)
        cond_model_input_ids = cond_model_input_ids.view(cond_model_input.shape[0], -1, model_input_ids.shape[-1])
        
        # Noise
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]
        
        # Timesteps
        # 简化：均匀采样
        indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=device)
        
        # Flow Matching
        sigmas = self._get_sigmas(timesteps, noise_scheduler, device)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
        
        # Pack & Concat
        packed_noisy_model_input = Flux2KleinPipeline._pack_latents(noisy_model_input)
        packed_cond_model_input = Flux2KleinPipeline._pack_latents(cond_model_input)
        
        packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_cond_model_input], dim=1)
        model_input_ids = torch.cat([model_input_ids, cond_model_input_ids], dim=1)
        
        # Prompt Embeds (简化：假设已预计算或此处计算)
        # 实际需调用 text_encoding_pipeline.encode_prompt
        prompt_embeds = batch.get("prompt_embeds", None) 
        # 此处省略文本编码细节，假设 batch 中已有或需外部传入
        
        # Transformer Forward
        guidance = torch.full([1], 1.5, device=device).expand(bsz) if transformer.config.guidance_embeds else None
        
        model_pred = transformer(
            hidden_states=packed_noisy_model_input,
            timestep=timesteps / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds, # 需确保维度正确
            txt_ids=None, # 需补充
            img_ids=model_input_ids,
            return_dict=False,
        )[0]
        
        # Unpack & Loss
        orig_input_shape = packed_noisy_model_input.shape
        model_pred = model_pred[:, :orig_input_shape[1], :]
        model_pred = Flux2KleinPipeline._unpack_latents_with_ids(model_pred, model_input_ids)
        
        loss = compute_flow_matching_loss(model_pred, noise, model_input, sigmas)
        return loss

    def _get_sigmas(self, timesteps, noise_scheduler, device):
        sigmas = noise_scheduler.sigmas.to(device=device, dtype=torch.float32)
        schedule_timesteps = noise_scheduler.timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < 4:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _get_latent_stats(self, vae):
        from models.flux.flux_klein import get_latent_stats
        return get_latent_stats(vae, self.accelerator.device)

    def _save_checkpoint(self, transformer, epoch):
        if self.accelerator.is_main_process:
            save_lora_checkpoint(transformer, self.config.training.output_dir, self.global_step)

    def _save_final_checkpoint(self, transformer):
        if self.accelerator.is_main_process:
            save_lora_checkpoint(transformer, self.config.training.output_dir, "final")