import torch
from diffusers import Flux2KleinPipeline
from registry.pipeline_registry import PipelineRegistry
from diffusers.training_utils import _collate_lora_metadata
from peft.utils import get_peft_model_state_dict

@PipelineRegistry.register('flux2kleinpipeline')
class Flux2kleinpipeline:
    def __init__(self,
                 pretrained_path=None,
                 vae=None,
                 transformer=None,
                 tokenizer=None,
                 text_encoder=None,
                 scheduler=None,
                 revision=None,
                 ):

        self.pretrained_path = pretrained_path
        self.vae = vae
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.revision = revision
        self._pipe = None

    @property
    def pipe(self):
        if self._pipe is None:
            self._pipe = Flux2KleinPipeline.from_pretrained(
                self.pretrained_path,
                vae=self.vae,
                transformer=self.transformer,
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                scheduler=self.scheduler,
                revision=self.revision,
            )
        return self._pipe
    @staticmethod
    def save_lora_weights(save_path, transformer):
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        modules_to_save = {"transformer": transformer}
        Flux2KleinPipeline.save_lora_weights(
            save_directory=save_path,
            transformer_lora_layers=transformer_lora_layers,
            **(_collate_lora_metadata(modules_to_save) if modules_to_save else {})
        )
