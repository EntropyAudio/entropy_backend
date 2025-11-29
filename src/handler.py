from pathlib import Path
import runpod
from .converter import extract_input
from .controller import run_inference
from .utils.utils import setup_logger
import entropy_training
import torch
from entropy_stable_audio_open.stable_audio_open import get_model
from entropy_training.src.utils.constants import CKPT_KEY_MODEL
from entropy_training.src.utils.utils import print_environment_info, set_cudnn_benchmarking, set_backend_precision, \
    adjust_ckpt_keys
from omegaconf import OmegaConf

cfg = OmegaConf.load(Path(entropy_training.__file__).parent / "configs" / "training.yaml")

print_environment_info()
set_cudnn_benchmarking(cfg)
set_backend_precision(cfg)

checkpoint = torch.load(Path(__file__).parent / "ckpt/checkpoint_1000.pth", map_location="cpu", weights_only=False)[CKPT_KEY_MODEL]
checkpoint = adjust_ckpt_keys(
    checkpoint,
    ignore_keys= {
        "conditioner.float_encoder_duration.embedder.embedding.0.weights",
        "conditioner.float_encoder_duration.embedder.embedding.1.weight",
        "conditioner.float_encoder_duration.embedder.embedding.1.bias",
    },
    replace_keys={
        "diffusion_model": "diffusion_transformer",
    }
)

model = get_model(cfg, download_pretrained_weights=False)
model.load_state_dict(checkpoint, strict=True)
model.to(cfg.environment.device)
model.eval()

def handler(event):
    logger = setup_logger()
    logger.info(f"Received event: {event}")

    prompt, batch_size = extract_input(event)

    output_list = run_inference(
        cfg=cfg,
        model=model,
        prompt=prompt,
        batch_size=batch_size,
    )

    return {
        "prompt": prompt,
        "audio_base64": output_list
    }

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
