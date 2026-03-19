import os
from pathlib import Path
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
from models.lora.inject import get_lora_state_dict
from diffusers.training_utils import _collate_lora_metadata

def save_lora_checkpoint(transformer, output_dir, step):
    save_path = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_path, exist_ok=True)
    
    transformer_lora_layers = get_lora_state_dict(transformer)
    modules_to_save = {"transformer": transformer}
    
    Flux2KleinPipeline.save_lora_weights(
        save_directory=save_path,
        transformer_lora_layers=transformer_lora_layers,
        **_collate_lora_metadata(modules_to_save),
    )
    print(f"Saved checkpoint to {save_path}")