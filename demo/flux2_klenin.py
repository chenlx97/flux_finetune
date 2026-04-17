import gradio as gr
import numpy as np
import random
import torch
import spaces
from diffusers import AutoModel, DiffusionPipeline, TorchAoConfig # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import Flux2KleinPipeline
from torchao.dtypes.affine_quantized_tensor import AffineQuantizedTensor
import argparse

def _safe_has_compatible_shallow_copy_type(t1, t2):
    return True
torch._has_compatible_shallow_copy_type = _safe_has_compatible_shallow_copy_type
AffineQuantizedTensor.__torch_function__ = torch._C._disabled_torch_function_impl


def create_pipe(base_weight: str, lora_weight: str, adapter_weights: float):

    model_dir = base_weight
    pipe = Flux2KleinPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16
        )
    pipe.load_lora_weights(lora_weight)
    pipe.set_adapters(["default_0"], adapter_weights=[adapter_weights])
    print(pipe.get_active_adapters())
    print("all:", pipe.get_list_adapters())
    pipe.to('cuda')
    return pipe

@spaces.GPU(duration=300)
def infer(
    image,
    prompt,
    seed=42,
    randomize_seed=False,
    guidance_scale=1.0,
    num_inference_steps=50,
):
    """
    Generates an image using the local Qwen-Image diffusers pipeline.
    """
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)
    print(f"Calling pipeline with prompt: '{prompt}'")
    print(f"Seed: {seed}, Steps: {num_inference_steps}")
    # if isinstance(image,list):
    #     image = [i[0].convert("RGB") for i in image]
    # else:
    #     image = image.convert("RGB")
    image = [image]
    image = pipe(
        # image=image if len(image) > 1 else image[0],
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        guidance_scale=guidance_scale,
    ).images[0]
    return image, seed


def create_ui():
    # --- Examples and UI Layout ---
    examples = []
    MAX_SEED = np.iinfo(np.int32).max
    css = """
    #col-container {
        margin: 0 auto;
        max-width: 1024px;
    }
    #edit_text{margin-top: -62px !important}
    """
    with gr.Blocks(css=css) as demo:
        with gr.Column(elem_id="col-container"):
            gr.Markdown("## 欢迎使用 Flux2.dev")
            with gr.Row():
                with gr.Column():
                    input_image = gr.Gallery(label="输入图片", show_label=False, type="pil",height=512)
                result = gr.Image(label="输出", show_label=False, type="pil",height=512,width=512)
            with gr.Row():
                prompt = gr.Text(
                        label="Prompt",
                        show_label=False,
                        placeholder="提示词",
                        container=False,
                )
                run_button = gr.Button("启动!", variant="primary")

            with gr.Accordion("采样设置", open=True):
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )

                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

                with gr.Row():

                    guidance_scale = gr.Slider(
                        label="指导率",
                        minimum=1.0,
                        maximum=10.0,
                        step=0.1,
                        value=4.0
                    )

                    num_inference_steps = gr.Slider(
                        label="采样步数",
                        minimum=1,
                        maximum=40,
                        step=1,
                        value=12,
                    )
        gr.on(
            triggers=[run_button.click, prompt.submit],
            fn=infer,
            inputs=[
                input_image,
                prompt,
                seed,
                randomize_seed,
                guidance_scale,
                num_inference_steps,
            ],
            outputs=[result, seed],
        )
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="/data/clx/control-lora-v2-master/ckpt/FLUX.2-klein-base-4B", help="base model path")
    parser.add_argument("--lora", type=str, default="/data/clx/tmp/test_1/checkpoint-1500/pytorch_lora_weights.safetensors",  help="lora weight path")
    parser.add_argument("--port", type=int, default=6102, help="server port")
    parser.add_argument("--adapter_weights", type=float, default=1.0, help="adapter weights")
    args = parser.parse_args()
    # --- device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # --- load model ---
    pipe = create_pipe(args.base_model, args.lora, args.adapter_weights)
    # --- constants ---
    MAX_SEED = np.iinfo(np.int32).max
    # --- launch UI ---
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=True
    )