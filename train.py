import argparse
import yaml
import torch
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
# 导入项目模块
from core.data.dataset import DreamBoothDataset, BucketBatchSampler, collate_fn
from registry.model_registry import ModelRegistry
from registry.trainer_registry import TrainerRegistry
from registry.pipeline_registry import PipelineRegistry
from core.adapters.lora  import setup_lora
from core.models.flux.flux2_klein import Flux2KleinModel 
from core.trainer.trainer import FluxTrainer
from core.pipeline.flux2kleinpipeline import Flux2kleinpipeline
from core.cache.textprecompute import TextPrecompute

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/flux2klein_lora.yaml")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # 将 dict 转为对象以便 trainer 使用
    class ConfigObj:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, ConfigObj(v))
                else:
                    setattr(self, k, v)
    full_config = ConfigObj(config)
    accelerator = Accelerator(
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        mixed_precision=config['model']['mixed_precision'],
    )
    
    # --- 使用 Registry 获取模型类 ---
    ModelCls = ModelRegistry.get(config['model']['model_name'])
    model_wrapper = ModelCls(
        pretrained_path=config['model']['pretrained_model_name_or_path'],
        dtype=torch.bfloat16 if config['model']['mixed_precision'] == 'bf16' else torch.float32,
        device=accelerator.device
    )
    transformer = model_wrapper.transformer
    vae = model_wrapper.vae
    text_encoder = model_wrapper.text_encoder
    tokenizer = model_wrapper.tokenizer
    scheduler = model_wrapper.scheduler
    model_wrapper.set_trainable(trainable=False) # 先全部冻结
    # --- 使用 Registry 获取训练器类 ---
    TrainerCls = TrainerRegistry.get(config['model']['trainer_name'])
    trainer = TrainerCls(accelerator, full_config)
    # --- 使用 Registry 获取文本编码 Pipeline ---
    PipelineCls = PipelineRegistry.get(config['model']['pipeline_name'])
    text_encoding_pipeline = PipelineCls(
        config['model']['pretrained_model_name_or_path'],
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
        
    ).pipe
    text_embeding = TextPrecompute(text_encoding_pipeline,config,device=accelerator.device).run()
    setup_lora(transformer, **config.get('lora', {}))
    for name, param in transformer.named_parameters():
        if "lora" in name:
            param.requires_grad = True
    
    # 训练样本尺寸设置
    buckets_str = config['data']['aspect_ratio_buckets']
    buckets = [tuple(map(int, b.split(','))) for b in buckets_str.split(';')]
    
    dataset = DreamBoothDataset(
        config=config,
        buckets=buckets,
        text_emb = text_embeding
    )

    batch_sampler = BucketBatchSampler(dataset, batch_size=config['data']['train_batch_size'], drop_last=True)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_sampler=batch_sampler, collate_fn=collate_fn,
        num_workers=config['data']['dataloader_num_workers']
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, transformer.parameters()),
        lr=config['training']['learning_rate'],
        betas=(config['training']['adam_beta1'], config['training']['adam_beta2']),
        weight_decay=config['training']['adam_weight_decay'],
    )
    
    # 准备
    transformer, optimizer, train_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader)
    
    trainer.train(
        train_dataloader=train_dataloader,
        transformer=transformer,
        optimizer=optimizer,
        lr_scheduler=get_scheduler(
            config['training']['lr_scheduler'],
            optimizer=optimizer,
            num_warmup_steps=config['training']['lr_warmup_steps'],
            num_training_steps=config['training']['max_train_steps']
        ),
        noise_scheduler=scheduler,
        vae=vae,
        text_encoding_pipeline=text_encoding_pipeline,
        validation_fn=None
    )

if __name__ == "__main__":
    main()