from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
def setup_lora(model_wrapper, rank=16, alpha=32,target_modules=None, dropout=0.0,logger=None):
    if target_modules is None:
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    model_wrapper.transformer.add_adapter(lora_config)

    trainable_parameters = [p for p in model_wrapper.transformer.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in trainable_parameters)
    total = sum(p.numel() for p in model_wrapper.transformer.parameters())
    if logger is not None:
        logger.info(f"Train lora modules: {target_modules}")
        logger.info(f"rank: {rank}, alpha: {alpha}, dropout: {dropout}")
        logger.info(f"Trainable parameters: {trainable}, Total: {total}, Ratio: {trainable/total:.6f}")
    
def get_lora_state_dict(transformer):
    
    return get_peft_model_state_dict(transformer)
