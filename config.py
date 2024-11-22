import enum

import torch
from pydantic import BaseModel, Field


class Device(enum.StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class Quantization(enum.StrEnum):
    DEFAULT = "default"
    FP16 = "fp16"
    INT8 = "int8"


class WhisperConfig(BaseModel):
    """See https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py#L599."""

    model: str = Field(default="Systran/faster-whisper-small")
    """
    Default Huggingface model to use for transcription. Note, the model must support being ran using CTranslate2.
    This model will be used if no model is specified in the request.

    Models created by authors of `faster-whisper` can be found at https://huggingface.co/Systran
    You can find other supported models at https://huggingface.co/models?p=2&sort=trending&search=ctranslate2 and https://huggingface.co/models?sort=trending&search=ct2
    """
    inference_device: Device = Device.CUDA if torch.cuda.is_available() else Device.CPU
    device_index: int | list[int] = 0
    compute_type: Quantization = Quantization.FP16 if torch.cuda.is_available() else Quantization.INT8
    cpu_threads: int = 0
    num_workers: int = 1
    ttl: int = Field(default=300, ge=-1)
    """
    Time in seconds until the model is unloaded if it is not being used.
    -1: Never unload the model.
    0: Unload the model immediately after usage.
    """
