from pathlib import Path
import runpod
from .converter import extract_input
from .controller import run_inference
from .utils.utils import setup_logger
import entropy_training
from entropy_stable_audio_open.stable_audio_open import get_model
# from entropy_training.utils.utils import environment_setup
from omegaconf import OmegaConf

cfg = OmegaConf.load(Path(entropy_training.__file__).parent / "config.yaml")
# environment_setup(cfg)
model = get_model(cfg, download_pretrained_weights=False).to(cfg.environment.device).eval()

def handler(event):
    logger = setup_logger()
    logger.info(f"Received event: {event}")
    input = extract_input(event)
    result = run_inference(cfg, input, model)
    return result

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
