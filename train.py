import argparse
import yaml
import torch
from accelerate import Accelerator
# 导入项目模块
from utils.logger import Logger
from registry.model_registry import ModelRegistry
from registry.trainer_registry import TrainerRegistry
from registry.dataset_registry import DatasetRegistry
from utils.training_utils import (
    prepare_text_embeddings,
    prepare_training_components,
    setup_model_adapter
)

def load_config(config_path):
    class ConfigObj:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, ConfigObj(v))
                else:
                    setattr(self, k, v)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        full_config = ConfigObj(config)
    return full_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/test.yaml")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.model.mixed_precision,
    )
    logger = Logger(config,accelerator)
    logger.info("Let's start!")

    # --- 使用 Registry 获取模型类 ---
    ModelCls = ModelRegistry.get(config.model.model_name)
    model_wrapper = ModelCls(
        config,
        dtype=torch.bfloat16 if config.model.mixed_precision == 'bf16' else torch.float32,
        device=accelerator.device
    )
    model_wrapper.set_trainable(trainable=False)
    logger.info("Model initialized")

    # --- 使用 Registry 获取训练器类 ---
    TrainerCls = TrainerRegistry.get(config.training.trainer_name)
    trainer = TrainerCls(accelerator, config,logger=logger)

    # --- 获取文本编码---
    text_embeding = prepare_text_embeddings(config,model_wrapper)    
    model_wrapper.to(accelerator.device)

    # --- 使用 Registry 获取数据类---
    DatasetCls = DatasetRegistry.get(config.data.data_type)
    dataset = DatasetCls(config,text_emb=text_embeding)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=DatasetCls.collate_fn,
        batch_size=config.training.train_batch_size,
        num_workers=config.data.dataloader_num_workers
    )
    # --- 优化器和调度器---
    setup_model_adapter(config, model_wrapper,logger)
    optimizer, train_dataloader, lr_scheduler = prepare_training_components(
        config, accelerator, model_wrapper,train_dataloader
    )
    trainer.train(
        train_dataloader=train_dataloader,
        model_wrapper=model_wrapper,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )

if __name__ == "__main__":
    main()