from .utils import constants as c
from .utils.exceptions import InvalidPromptError
import logging

logger = logging.getLogger(c.LOGGER_NAME)

def extract_input(event):
    logger.info(f"Extracting and validating input...")

    input = event[c.INPUT]
    prompt = input.get(c.PROMPT)

    if not prompt or not isinstance(prompt, str) or prompt == "":
        raise InvalidPromptError(prompt=prompt)

    return input
