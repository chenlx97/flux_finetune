import argparse
import yaml
import torch
from accelerate import Accelerator
from pathlib import Path

# 导入项目模块
from data.dataset import DreamBoothDataset, BucketBatchSampler, collate_fn
from models.flux.flux_klein import load_flux_components
from models.lora.inject import setup_lora
from trainer.trainer import FluxTrainer
from registry.trainer_registry import TrainerRegistry

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--lora_config", type=str, default="configs/lora.yaml")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    lora_config = load_config(args.lora_config)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        mixed_precision=config['model']['mixed_precision'],
    )
    
    # 加载模型
    components = load_flux_components(
        config['model']['pretrained_model_name_or_path'],
        dtype=torch.bfloat16 if config['model']['mixed_precision'] == 'bf16' else torch.float32,
        device=accelerator.device
    )
    
    transformer = components['transformer']
    vae = components['vae']
    text_encoder = components['text_encoder']
    tokenizer = components['tokenizer']
    scheduler = components['scheduler']
    
    # 冻结非训练参数
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # 注入 LoRA
    setup_lora(transformer, **lora_config['lora'])
    
    # 数据集
    # 解析 buckets 字符串
    buckets_str = config['data']['aspect_ratio_buckets']
    buckets = [tuple(map(int, b.split(','))) for b in buckets_str.split(';')]
    
    dataset = DreamBoothDataset(
        instance_data_root=config['data']['instance_data_dir'],
        instance_prompt=config['validation']['validation_prompt'],
        buckets=buckets,
        resolution=config['data']['resolution'],
        center_crop=config['data']['center_crop'],
        random_flip=config['data']['random_flip'],
    )
    
    batch_sampler = BucketBatchSampler(dataset, batch_size=config['data']['train_batch_size'], drop_last=True)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_sampler=batch_sampler, collate_fn=collate_fn,
        num_workers=config['data']['dataloader_num_workers']
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=config['training']['learning_rate'],
        betas=(config['training']['adam_beta1'], config['training']['adam_beta2']),
        weight_decay=config['training']['adam_weight_decay'],
    )
    
    # 准备
    transformer, optimizer, train_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader)
    
    # 训练器
    trainer = FluxTrainer(accelerator, type('Config', (object,), config)())
    
    # 这里需要构建 text_encoding_pipeline 用于训练中的 prompt 编码，简化起见略过详细实现
    text_encoding_pipeline = None 
    
    trainer.train(
        train_dataloader=train_dataloader,
        transformer=transformer,
        optimizer=optimizer,
        lr_scheduler=None, # 需初始化 scheduler
        noise_scheduler=scheduler,
        vae=vae,
        text_encoding_pipeline=text_encoding_pipeline,
        validation_fn=None
    )

if __name__ == "__main__":
    main()