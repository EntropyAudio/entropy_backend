import pytest
import torch
import torchaudio
import base64
import io
from unittest.mock import MagicMock
from src.controller import run_inference
from entropy_data.src.dataset.models import AudioConditioning

class MockConfig:
    def __init__(self):
        self.environment = type('env', (), {'device': 'cpu'})()
        self.audio = type('audio', (), {
            'latent_size': (128, 128),
            'sample_rate': 44100
        })()

def test_run_inference_happy_path(mocker):
    mock_cfg = MockConfig()
    test_input = {"prompt": "a test prompt"}
    mock_model = MagicMock()
    mock_output_tensor = torch.randn(1, 1, 44100)
    mock_model.generate.return_value = mock_output_tensor

    mocker.patch(
        'entropy_data.src.dataset.audio_utils.trim_silence',
        side_effect=lambda x: x
    )

    result_b64 = run_inference(mock_cfg, test_input, mock_model)

    mock_model.generate.assert_called_once()
    args, kwargs = mock_model.generate.call_args
    conditioning_arg = kwargs['conditioning'][0]
    assert isinstance(conditioning_arg, AudioConditioning)
    assert conditioning_arg.description == "a test prompt"

    assert isinstance(result_b64, str)
    decoded_wav = base64.b64decode(result_b64)
    buffer = io.BytesIO(decoded_wav)
    buffer.seek(0)
    loaded_tensor, sample_rate = torchaudio.load(buffer)

    assert sample_rate == mock_cfg.audio.sample_rate
    assert loaded_tensor.shape == (1, 44100)
    assert loaded_tensor.dtype == torch.float32
