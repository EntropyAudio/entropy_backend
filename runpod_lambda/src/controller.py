import os
import logging
import torch
import torchaudio
from einops import rearrange
from entropy_data.src.dataset.audio_utils import trim_silence
from entropy_data.src.dataset.models import AudioConditioning
from .utils import constants as c

logger = logging.getLogger(c.LOGGER_NAME)

def run_inference(cfg, input, model):
    logger.info("Running inference...")

    with torch.inference_mode() and torch.autocast(device_type=cfg.environment.device, dtype=eval(cfg.training.dtype)):
        output = model.generate(
            steps=100,
            cfg_scale=7.0,
            conditioning=[AudioConditioning(cfg=cfg, inference=True, description=input["prompt"])],
            latent_size=cfg.audio.latent_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
        )

        output = rearrange(output, "b d n -> d (b n)")
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        output = trim_silence(output)
        # return output

        filename = f"{input['prompt']}.wav"
        save_dir = f"./{cfg.demo.path}"
        full_path = f"./{cfg.demo.path}/{filename}"
        os.makedirs(save_dir, exist_ok=True)
        torchaudio.save(full_path, output, cfg.audio.sample_rate)
