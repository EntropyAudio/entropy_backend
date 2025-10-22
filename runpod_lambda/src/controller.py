import logging
from .utils import constants as c

logger = logging.getLogger(c.LOGGER_NAME)

def run_inference(input, diffusion_trainer):
    logger.info("Running inference...")
    print(diffusion_trainer.model)
