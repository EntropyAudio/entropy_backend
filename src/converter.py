from .utils import constants as c
from .utils.exceptions import InvalidPromptError, InvalidRequestError
import logging

logger = logging.getLogger(c.LOGGER_NAME)

def extract_input(event):
    logger.info(f"Extracting and validating input...")
    if "input" not in event:
        raise InvalidRequestError(request=event)

    input = event.get("input")
    prompt = input.get("prompt")

    if not prompt or not isinstance(prompt, str) or prompt == "":
        raise InvalidPromptError(prompt=prompt)

    return input
