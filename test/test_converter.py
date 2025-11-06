import pytest
from src.converter import extract_input
from src.utils.exceptions import InvalidPromptError, InvalidRequestError

def test_extract_input_valid():
    event = {
        "input": {
            "prompt": "This is a test prompt."
        }
    }
    result = extract_input(event)
    assert result == event["input"]

@pytest.mark.parametrize("invalid_prompt", [
    None,        # Prompt is None
    "",          # Prompt is an empty string
    123,         # Prompt is not a string (an integer)
    [],          # Prompt is not a string (a list)
    {}           # Prompt is not a string (a dictionary)
])
def test_extract_input_invalid_prompt(invalid_prompt):
    event = {
        "input": {
            "prompt": invalid_prompt
        }
    }
    with pytest.raises(InvalidPromptError):
        extract_input(event)

def test_extract_input_missing_prompt_key():
    event = {
        "input": {
            "some_other_key": "some_value"
        }
    }
    with pytest.raises(InvalidPromptError):
        extract_input(event)

def test_extract_input_missing_input_key():
    event = {
        "another_key": "some_value"
    }
    with pytest.raises(InvalidRequestError):
        extract_input(event)
