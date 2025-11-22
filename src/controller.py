import io
import os
import logging
import torch
import torchaudio
from einops import rearrange
from entropy_data.src.dataset.audio_utils import trim_silence
from entropy_data.src.dataset.models import AudioConditioning
from .utils import constants as c
import base64

logger = logging.getLogger(c.LOGGER_NAME)

def run_inference(cfg, model, prompt, batch_size):
    logger.info("Running inference...")

    with torch.inference_mode() and torch.autocast(device_type=cfg.environment.device, dtype=torch.float32):
        output = model.generate(
            steps=140,
            cfg_scale=6.0,
            conditioning=[
                AudioConditioning(cfg=cfg, inference=True, description=prompt, key=None, bpm=None, loop=None) for i in range(batch_size)
            ],
            latent_size=cfg.audio.latent_size,
            sigma_min=0.3,
            sigma_max=500,
            # rho=1.0,
            sampler_type="dpmpp-3m-sde",
            # sampler_type="dpmpp-2m-sde",
            # sampler_type="dpmpp-2m",
            # sampler_type="k-dpmpp-2s-ancestral",
        )

        max_vals = torch.max(torch.abs(output), dim=-1, keepdim=True)[0] + 1e-7

        output = (
            output
            .div(max_vals)
            .clamp(-1, 1)
            .mul(32767)
            .round()
            .to(torch.int16)
        )

        output_list = []
        for audio_tensor in output.cpu():
            trimmed_audio = trim_silence(audio_tensor)

            buffer = io.BytesIO()
            torchaudio.save(
                buffer,
                trimmed_audio,
                cfg.audio.sample_rate,
                format="wav"
            )
            buffer.seek(0)
            base64_audio = base64.b64encode(buffer.read()).decode('utf-8')
            output_list.append(base64_audio)

        return output_list
