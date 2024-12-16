import enum
from typing import List

import torch
from pydantic import BaseModel, Field

from api_models.enums import TimestampGranularity, ResponseFormat


class Device(enum.StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class Quantization(enum.StrEnum):
    INT8 = "int8"
    INT8_FLOAT16 = "int8_float16"
    INT8_BFLOAT16 = "int8_bfloat16"
    INT8_FLOAT32 = "int8_float32"
    INT16 = "int16"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    DEFAULT = "default"


class WhisperModelConfig(BaseModel):
    """See https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py#L599."""

    inference_device: Device = Device.CUDA if torch.cuda.is_available() else Device.AUTO
    device_index: int | list[int] = 0
    compute_type: Quantization = Quantization.FLOAT16 if torch.cuda.is_available() else Quantization.DEFAULT
    cpu_threads: int = 0
    num_workers: int = 1
    ttl: int = Field(default=300, ge=-1)
    """
    Time in seconds until the model is unloaded if it is not being used.
    -1: Never unload the model.
    0: Unload the model immediately after usage.
    """


class FasterWhisperConfig(BaseModel):
    default_model_name: str = "large-v3"
    default_prompt: str = None
    default_language: str = None
    default_response_format: ResponseFormat = ResponseFormat.JSON
    default_temperature: float = 0.0
    default_timestamp_granularities: List[TimestampGranularity] = [TimestampGranularity.SEGMENT]

    min_temperature: float = 0.0
    max_temperature: float = 2.0


faster_whisper_config = FasterWhisperConfig()
