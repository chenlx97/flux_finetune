from peft import LoraConfig

def setup_lora(model, rank=16, alpha=32, dropout=0.0, target_modules=None):
    if target_modules is None:
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    model.add_adapter(lora_config)
    return model

def get_lora_state_dict(model):
    from peft.utils import get_peft_model_state_dict
    return get_peft_model_state_dict(model)