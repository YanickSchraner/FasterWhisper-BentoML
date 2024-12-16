from pathlib import Path
from typing import Annotated, List, Union, Optional

from bentoml.exceptions import InvalidArgument
from huggingface_hub import ModelInfo
from pydantic import Field, BeforeValidator, BaseModel, confloat

from api_models.enums import TimestampGranularity, ResponseFormat, Language
from api_models.output_models import ModelObject, logger
from config import faster_whisper_config

ModelName = Annotated[
    str,
    Field(
        description="The ID of the model. You can get a list of available models by calling `/v1/models`.",
        examples=[
            "Systran/faster-distil-whisper-large-v3",
            "bofenghuang/whisper-large-v2-cv11-french-ct2",
        ],
    ),
]


def _convert_timestamp_granularities(timestamp_granularities: str | List[TimestampGranularity]) -> List[
    TimestampGranularity]:
    if isinstance(timestamp_granularities, List):
        return timestamp_granularities

    timestamps = timestamp_granularities.split(",")
    return [TimestampGranularity(t.strip()) for t in timestamps]


TimestampGranularities = Annotated[
    List[TimestampGranularity],
    BeforeValidator(_convert_timestamp_granularities)
]


def _convert_temperature(temperature: Union[str, int, float, List[float]]) -> List[float]:
    if isinstance(temperature, List):
        return temperature
    elif isinstance(temperature, (float, int)):
        return [float(temperature)]

    temperatures = temperature.split(",")

    return [float(t.strip()) for t in temperatures]


BoundedTemperature = confloat(ge=faster_whisper_config.min_temperature, le=faster_whisper_config.max_temperature)

ValidatedTemperature = Annotated[
    Union[BoundedTemperature, List[BoundedTemperature]],
    BeforeValidator(_convert_temperature),
]


def _process_empty_response_format(response_format: Optional[str]) -> Optional[str]:
    if response_format == '':
        return faster_whisper_config.default_response_format
    return response_format


ValidatedResponseFormat = Annotated[
    ResponseFormat,
    BeforeValidator(_process_empty_response_format)
]


def _process_empty_language(language: Optional[str]) -> Optional[str]:
    if language == '':
        return faster_whisper_config.default_language
    return language


# None needed to return None from Validator
ValidatedLanguage = Annotated[
    Language | None,
    BeforeValidator(_process_empty_language)
]


def hf_model_info_to_model_object(model: ModelInfo) -> ModelObject:
    assert model.created_at is not None
    assert model.card_data is not None
    assert model.card_data.language is None or isinstance(
        model.card_data.language, str | list
    )
    if model.card_data.language is None:
        language = []
    elif isinstance(model.card_data.language, str):
        language = [model.card_data.language]
    else:
        language = model.card_data.language
    transformed_model = ModelObject(
        id=model.id,
        created=int(model.created_at.timestamp()),
        object_="model",
        owned_by=model.id.split("/")[0],
        language=language,
    )
    return transformed_model


def validate_timestamp_granularities(response_format, timestamp_granularities):
    if (
            timestamp_granularities != faster_whisper_config.default_timestamp_granularities
            and response_format != ResponseFormat.VERBOSE_JSON
    ):
        logger.warning(
            "It only makes sense to provide `timestamp_granularities[]` when `response_format` is set to "
            "`verbose_json`. See https://platform.openai.com/docs/api-reference/audio/createTranscription#audio"
            "-createtranscription-timestamp_granularities."
            # noqa: E501
        )

    if (
            "word" not in timestamp_granularities
            and response_format == ResponseFormat.VERBOSE_JSON
    ):
        raise InvalidArgument(
            f"timestamp_granularities must contain 'word' when response_format "
            f"is set to {ResponseFormat.VERBOSE_JSON}"
        )


class BatchTranscriptionRequest(BaseModel):
    file: Path
    model: Optional[str] = faster_whisper_config.default_model_name
    language: Optional[str] = faster_whisper_config.default_language
    prompt: Optional[str] = faster_whisper_config.default_prompt
    response_format: Optional[ResponseFormat] = faster_whisper_config.default_response_format
    temperature: Optional[Union[float, List[float]]] = faster_whisper_config.default_temperature
    timestamp_granularities: Optional[TimestampGranularities] = faster_whisper_config.default_timestamp_granularities
