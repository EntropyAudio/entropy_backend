import io
import os
import logging
import torch
import torchaudio
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
                AudioConditioning(cfg=cfg, inference=True, description=prompt, key=None, bpm=None, loop=None) for _ in range(batch_size)
            ],
            latent_size=cfg.audio.latent_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
        )

        output_list = []
        output_cpu = output.cpu()

        for idx, audio_tensor in enumerate(output_cpu):
            trimmed_audio = trim_silence(audio_tensor, threshold=0.001)

            max_val = torch.max(torch.abs(trimmed_audio)) + 1e-7
            norm_audio = trimmed_audio / max_val
            final_audio = (norm_audio.clamp(-1, 1).mul(32767).round().to(torch.int16))

            # save_dir = "./outputs"
            # os.makedirs(save_dir, exist_ok=True)
            # file_path = os.path.join(save_dir, f"output_{idx}.wav")
            # torchaudio.save(
            #     file_path,
            #     final_audio,
            #     cfg.audio.sample_rate,
            #     format="wav"
            # )

            buffer = io.BytesIO()
            torchaudio.save(
                buffer,
                final_audio,
                cfg.audio.sample_rate,
                format="wav"
            )
            buffer.seek(0)
            base64_audio = base64.b64encode(buffer.read()).decode('utf-8')
            output_list.append(base64_audio)

        return output_list