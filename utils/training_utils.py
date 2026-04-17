import torch
from diffusers.optimization import get_scheduler
from core.adapters.lora import setup_lora
from core.cache.textprecompute import TextPrecompute

def prepare_text_embeddings(config, model_wrapper):
    text_embeding = TextPrecompute(model_wrapper, config)
    text_embeding.run()
    return text_embeding

def prepare_training_components(config, accelerator, model_wrapper, train_dataloader):

    if config.model.model_adapter == 'lora':
        model = model_wrapper.transformer
        target_attr = "transformer"
    elif config.model.model_adapter == 'controlnet':
        model = model_wrapper.controlnet
        target_attr = "controlnet"
    else:
        raise ValueError("Unknown model_adapter")
    
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.training.learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        weight_decay=config.training.adam_weight_decay,
    )
    lr_scheduler = get_scheduler(
        config.training.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.training.lr_warmup_rate * config.training.max_train_steps,
    )
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        lr_scheduler
    )
    setattr(model_wrapper, target_attr, model)

    return optimizer, train_dataloader, lr_scheduler

def setup_model_adapter(config, model_wrapper,logger=None):
    if config.model.model_adapter == 'lora':
        setup_lora(model_wrapper, config.lora.rank, config.lora.alpha, config.lora.target_modules, config.lora.dropout,logger)
    # if 'controlnet' in config:
    #     setup_controlnet(model, **config.controlnet)
    
