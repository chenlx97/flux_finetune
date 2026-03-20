import os
import torch
from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
from registry.trainer_registry import TrainerRegistry
from core.adapters.lora import get_lora_state_dict
from diffusers.training_utils import _collate_lora_metadata
from registry.pipeline_registry import PipelineRegistry

@TrainerRegistry.register('flux2_lora')
class FluxTrainer:
    def __init__(self, accelerator: Accelerator, config):
        self.accelerator = accelerator
        self.config = config
        self.global_step = 0
        
    def train(self, train_dataloader, transformer, optimizer, lr_scheduler, noise_scheduler, 
              vae, text_encoding_pipeline, validation_fn=None):
        transformer.train()
        progress_bar = tqdm(range(self.config.training.max_train_steps), disable=not self.accelerator.is_local_main_process)
        
        # 获取 VAE 统计量
        latents_bn_mean, latents_bn_std = self._get_latent_stats(vae)
        
        for epoch in range(self.config.training.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate([transformer]):
                    loss = self._train_step(
                        batch, transformer, vae, text_encoding_pipeline, 
                        noise_scheduler, latents_bn_mean, latents_bn_std, self.config
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

    def _train_step(self, batch, transformer, vae, text_encoding_pipeline, noise_scheduler, 
                    latents_bn_mean, latents_bn_std, config):
        device = self.accelerator.device
        pixel_values = batch["pixel_values"].to(device)
        cond_pixel_values = batch["cond_pixel_values"].to(device)
        prompts = batch["prompts"]
        
        # 1. Encode latents
        model_input = vae.encode(pixel_values).latent_dist.mode()
        cond_model_input = vae.encode(cond_pixel_values).latent_dist.mode()
        
        # 2. Patchify & Norm (Flux2 Klein 特有逻辑)
        model_input = Flux2KleinPipeline._patchify_latents(model_input)
        model_input = (model_input - latents_bn_mean) / latents_bn_std
        
        cond_model_input = Flux2KleinPipeline._patchify_latents(cond_model_input)
        cond_model_input = (cond_model_input - latents_bn_mean) / latents_bn_std
        
        # 3. Prepare IDs
        model_input_ids = Flux2KleinPipeline._prepare_latent_ids(model_input).to(device=device)
        cond_model_input_list = [cond_model_input[i].unsqueeze(0) for i in range(cond_model_input.shape[0])]
        cond_model_input_ids = Flux2KleinPipeline._prepare_image_ids(cond_model_input_list).to(device=cond_model_input.device)
        cond_model_input_ids = cond_model_input_ids.view(cond_model_input.shape[0], -1, model_input_ids.shape[-1])
        
        # 4. Noise & Timesteps
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]
        
        u = compute_density_for_timestep_sampling(
            weighting_scheme=config.training.weighting_scheme,
            batch_size=bsz,
            logit_mean=config.training.logit_mean,
            logit_std=config.training.logit_std,
            mode_scale=config.training.mode_scale,
        )
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=device)
        
        # 5. Flow Matching
        sigmas = self._get_sigmas(timesteps, noise_scheduler, device)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
        
        # 6. Pack & Concat
        packed_noisy_model_input = Flux2KleinPipeline._pack_latents(noisy_model_input)
        packed_cond_model_input = Flux2KleinPipeline._pack_latents(cond_model_input)
        
        orig_input_shape = packed_noisy_model_input.shape
        orig_input_ids_shape = model_input_ids.shape
        
        packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_cond_model_input], dim=1)
        model_input_ids = torch.cat([model_input_ids, cond_model_input_ids], dim=1)
        
        # 7. Prompt Embeds (简化：假设 batch 中已有或需外部传入，此处复用原逻辑)
        # 实际需调用 text_encoding_pipeline.encode_prompt，为简洁此处假设 prompt_embeds 已准备好
        # 在实际 train.py 中需要处理 text_encoding
        prompt_embeds = batch.get("prompt_embeds", None) 
        text_ids = batch.get("text_ids", None)
        
        # 8. Transformer Forward
        guidance = torch.full([1], config.training.guidance_scale, device=device).expand(bsz) if transformer.config.guidance_embeds else None
        
        model_pred = transformer(
            hidden_states=packed_noisy_model_input,
            timestep=timesteps / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=model_input_ids,
            return_dict=False,
        )[0]
        
        # 9. Unpack & Loss
        model_pred = model_pred[:, :orig_input_shape[1], :]
        model_input_ids = model_input_ids[:, :orig_input_ids_shape[1], :]
        model_pred = Flux2KleinPipeline._unpack_latents_with_ids(model_pred, model_input_ids)
        
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=config.training.weighting_scheme, sigmas=sigmas)
        target = noise - model_input
        
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        return loss.mean()

    def _get_sigmas(self, timesteps, noise_scheduler, device):
        sigmas = noise_scheduler.sigmas.to(device=device, dtype=torch.float32)
        schedule_timesteps = noise_scheduler.timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < 4:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _get_latent_stats(self, vae):
        if hasattr(vae, 'bn'):
            latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(self.accelerator.device)
            latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(self.accelerator.device)
        else:
            latents_bn_mean = torch.zeros(1, 1, 1, 1).to(self.accelerator.device)
            latents_bn_std = torch.ones(1, 1, 1, 1).to(self.accelerator.device)
        return latents_bn_mean, latents_bn_std

    def _save_checkpoint(self, transformer, epoch):
        if self.accelerator.is_main_process:
            self._save_lora(transformer, f"checkpoint-{self.global_step}")

    def _save_final_checkpoint(self, transformer):
        if self.accelerator.is_main_process:
            self._save_lora(transformer, "final")
            
    def _save_lora(self, transformer, name):
        save_path = os.path.join(self.config.training.output_dir, name)
        os.makedirs(save_path, exist_ok=True)
        PipelineCls = PipelineRegistry.get(self.config['model']['pipeline_name'])
        PipelineCls().save_lora_weights(save_path, transformer)
        print(f"Saved checkpoint to {save_path}")

        