import logging
from typing import TYPE_CHECKING, Annotated, AsyncGenerator, List, Optional, Union

import bentoml
import huggingface_hub
from bentoml.validators import ContentType
from fastapi import FastAPI

from api_models import (
    Language,
    ModelListResponse,
    ModelObject,
    ResponseFormat,
    DEFAULT_TIMESTAMP_GRANULARITIES, TimestampGranularity,
    segments_to_response,
    segments_to_streaming_response,
    validate_timestamp_granularities, BatchTranscriptionRequest, ModelName
)
from config import WhisperConfig
from core import Segment
from logger import configure_logging
from model_manager import WhisperModelManager

if TYPE_CHECKING:
    from huggingface_hub.hf_api import ModelInfo

from http import HTTPStatus
from pathlib import Path
from pydantic import Field

from bentoml.exceptions import NotFound

logger = logging.getLogger(__name__)

fastapi = FastAPI()

configure_logging()


@bentoml.service(
    traffic={"timeout": 3000},
    resources={
        "gpu": 1,
        "memory": "8Gi",
    },
)
class FasterWhisper:
    def __init__(self):
        self.model_manager = WhisperModelManager(WhisperConfig())

    @bentoml.api(route="/v1/audio/transcriptions")
    def transcribe(
            self,
            file: Annotated[Path, ContentType("audio/mpeg")],
            model: Optional[ModelName] = Field(
                default="large-v3", description="Whisper model to load"
            ),
            language: Optional[Language] = Field(
                default=None,
                description='The language spoken in the audio. It should be a language code such as "en" or "fr". If '
                            'not set, the language will be detected in the first 30 seconds of audio.',
            ),
            prompt: Optional[float] = Field(
                default=None,
                description="Optional text string or iterable of token ids to provide as a prompt for the first window.",
            ),
            response_format: Optional[ResponseFormat] = Field(
                default=ResponseFormat.JSON,
                description=f"One of: {[format for format in ResponseFormat]}",
            ),
            temperature: Optional[Union[float, List[float]]] = Field(
                default=0.0,
                description="Temperature value, which can either be a single float or a list of floats.",
            ),
            timestamp_granularities: Optional[List[TimestampGranularity]] = Field(
                default=DEFAULT_TIMESTAMP_GRANULARITIES,
                alias="timestamp_granularities[]",
                description="The timestamp granularities to populate for this transcription. response_format must be "
                            "set verbose_json to use timestamp granularities."
            )
    ) -> (
            Annotated[str, bentoml.validators.ContentType("text/plain")]
            | Annotated[str, bentoml.validators.ContentType("application/json")]
            | Annotated[str, bentoml.validators.ContentType("text/vtt")]
            | Annotated[str, bentoml.validators.ContentType("text/event-stream")]
    ):
        return self._transcribe_audio(file,
                                      model,
                                      language,
                                      prompt,
                                      response_format,
                                      temperature,
                                      timestamp_granularities)

    @bentoml.api(
        batchable=True,
        max_batch_size=4,
        max_latency_ms=60000,
        route="/v1/audio/transcriptions/batch"
    )
    async def batch_transcribe(self, requests: List[BatchTranscriptionRequest]) -> List[str]:
        print(f"number of requests processed: {len(requests)}")
        return [self._transcribe_audio(request.file,
                                       request.model,
                                       request.language,
                                       request.prompt,
                                       request.response_format,
                                       request.temperature,
                                       request.timestamp_granularities)
                for request in requests]

    @bentoml.task(
        route="/v1/audio/transcriptions/task")
    def task_transcribe(
            self,
            file: Annotated[Path, ContentType("audio/mpeg")],
            model: Optional[ModelName] = Field(
                default="large-v3", description="Whisper model to load"
            ),
            language: Optional[Language] = Field(
                default=None,
                description='The language spoken in the audio. It should be a language code such as "en" or "fr". If '
                            'not set, the language will be detected in the first 30 seconds of audio.',
            ),
            prompt: Optional[float] = Field(
                default=None,
                description="Optional text string or iterable of token ids to provide as a prompt for the first window.",
            ),
            response_format: Optional[ResponseFormat] = Field(
                default=ResponseFormat.JSON,
                description=f"One of: {[format for format in ResponseFormat]}",
            ),
            temperature: Optional[Union[float, List[float]]] = Field(
                default=0.0,
                description="Temperature value, which can either be a single float or a list of floats.",
            ),
            timestamp_granularities: Optional[List[TimestampGranularity]] = Field(
                default=DEFAULT_TIMESTAMP_GRANULARITIES,
                alias="timestamp_granularities[]",
                description="The timestamp granularities to populate for this transcription. response_format must be "
                            "set verbose_json to use timestamp granularities."
            )
    ) -> (
            Annotated[str, bentoml.validators.ContentType("text/plain")]
            | Annotated[str, bentoml.validators.ContentType("application/json")]
            | Annotated[str, bentoml.validators.ContentType("text/vtt")]
            | Annotated[str, bentoml.validators.ContentType("text/event-stream")]
    ):
        return self._transcribe_audio(file,
                                      model,
                                      language,
                                      prompt,
                                      response_format,
                                      temperature,
                                      timestamp_granularities)

    @bentoml.api(route="/v1/audio/transcriptions/stream")
    async def streaming_transcribe(
            self,
            file: Annotated[Path, ContentType("audio/mpeg")],
            model: Optional[ModelName] = Field(
                default="large-v3", description="Whisper model to load"
            ),
            language: Optional[Language] = Field(
                default=None,
                description='The language spoken in the audio. It should be a language code such as "en" or "fr". If '
                            'not set, the language will be detected in the first 30 seconds of audio.',
            ),
            prompt: Optional[float] = Field(
                default=None,
                description="Optional text string or iterable of token ids to provide as a prompt for the first window.",
            ),
            response_format: Optional[ResponseFormat] = Field(
                default=ResponseFormat.JSON,
                description=f"One of: {[format for format in ResponseFormat]}",
            ),
            temperature: Optional[Union[float, List[float]]] = Field(
                default=0.0,
                description="Temperature value, which can either be a single float or a list of floats.",
            ),
            timestamp_granularities: Optional[List[TimestampGranularity]] = Field(
                default=DEFAULT_TIMESTAMP_GRANULARITIES,
                alias="timestamp_granularities[]",
                description="The timestamp granularities to populate for this transcription. response_format must be "
                            "set verbose_json to use timestamp granularities."
            )
    ) -> AsyncGenerator[str, None]:

        validate_timestamp_granularities(response_format, timestamp_granularities)

        segments, transcription_info = self._prepare_audio_segments(file, language, model, prompt, temperature,
                                                                    timestamp_granularities)
        generator = segments_to_streaming_response(
            segments, transcription_info, response_format
        )

        for chunk in generator:
            yield chunk

    @fastapi.get("/v1/models")
    def get_models(self) -> ModelListResponse:
        models = huggingface_hub.list_models(
            library="ctranslate2", tags="automatic-speech-recognition", cardData=True
        )
        models = list(models)
        models.sort(key=lambda model: model.downloads, reverse=True)  # type: ignore  # noqa: PGH003
        transformed_models: list[ModelObject] = []
        for model in models:
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
            transformed_models.append(transformed_model)
        return ModelListResponse(data=transformed_models)

    @fastapi.get("/v1/models/{model_name:path}")
    # NOTE: `examples` doesn't work https://github.com/tiangolo/fastapi/discussions/10537
    def get_model(self,
                  model_name: Annotated[
                      str, Path(example="Systran/faster-distil-whisper-large-v3")
                  ],
                  ) -> ModelObject:
        models = huggingface_hub.list_models(
            model_name=model_name,
            library="ctranslate2",
            tags="automatic-speech-recognition",
            cardData=True,
        )
        models = list(models)
        models.sort(key=lambda model: model.downloads, reverse=True)  # type: ignore  # noqa: PGH003
        if len(models) == 0:
            raise NotFound(
                error_code=HTTPStatus.NOT_FOUND, detail="Model doesn't exists"
            )
        exact_match: ModelInfo | None = None
        for model in models:
            if model.id == model_name:
                exact_match = model
                break
        if exact_match is None:
            raise NotFound(
                error_code=HTTPStatus.NOT_FOUND,
                detail=f"Model doesn't exists. Possible matches: {', '.join([model.id for model in models])}",
            )
        assert exact_match.created_at is not None
        assert exact_match.card_data is not None
        assert exact_match.card_data.language is None or isinstance(
            exact_match.card_data.language, str | list
        )
        if exact_match.card_data.language is None:
            language = []
        elif isinstance(exact_match.card_data.language, str):
            language = [exact_match.card_data.language]
        else:
            language = exact_match.card_data.language
        return ModelObject(
            id=exact_match.id,
            created=int(exact_match.created_at.timestamp()),
            object_="model",
            owned_by=exact_match.id.split("/")[0],
            language=language,
        )

    def _transcribe_audio(self,
                          file,
                          model,
                          language,
                          prompt,
                          response_format,
                          temperature,
                          timestamp_granularities):
        validate_timestamp_granularities(response_format, timestamp_granularities)

        segments, transcription_info = self._prepare_audio_segments(file, language, model,
                                                                    prompt, temperature,
                                                                    timestamp_granularities)
        return segments_to_response(segments, transcription_info, response_format)

    def _prepare_audio_segments(self, file, language, model, prompt, temperature, timestamp_granularities):
        with self.model_manager.load_model(model) as whisper:
            segments, transcription_info = whisper.transcribe(
                file,
                initial_prompt=prompt,
                language=language,
                temperature=temperature,
                word_timestamps="word" in timestamp_granularities,
            )
        segments = Segment.from_faster_whisper_segments(segments)
        return segments, transcription_info


# according to their docs: https://docs.bentoml.org/en/latest/get-started/adaptive-batching.html#handle-multiple
# -parameters
@bentoml.service(
    traffic={"timeout": 3000},
)
class WrapperFasterWhisper:
    batch = bentoml.depends(FasterWhisper)

    @bentoml.api(route="/v1/audio/transcriptions/batch")
    async def wrap_transcribe(
            self,
            file: Annotated[Path, ContentType("audio/mpeg")],
            model: Optional[ModelName] = Field(
                default="large-v3", description="Whisper model to load"
            ),
            language: Optional[Language] = Field(
                default=None,
                description='The language spoken in the audio. It should be a language code such as "en" or "fr". If '
                            'not set, the language will be detected in the first 30 seconds of audio.',
            ),
            prompt: Optional[float] = Field(
                default=None,
                description="Optional text string or iterable of token ids to provide as a prompt for the first window.",
            ),
            response_format: Optional[ResponseFormat] = Field(
                default=ResponseFormat.JSON,
                description=f"One of: {[format for format in ResponseFormat]}",
            ),
            temperature: Optional[Union[float, List[float]]] = Field(
                default=0.0,
                description="Temperature value, which can either be a single float or a list of floats.",
            ),
            timestamp_granularities: Optional[List[TimestampGranularity]] = Field(
                default=DEFAULT_TIMESTAMP_GRANULARITIES,
                alias="timestamp_granularities[]",
                description="The timestamp granularities to populate for this transcription. response_format must be "
                            "set verbose_json to use timestamp granularities."
            )) -> (
            Annotated[str, bentoml.validators.ContentType("text/plain")]
            | Annotated[str, bentoml.validators.ContentType("application/json")]
            | Annotated[str, bentoml.validators.ContentType("text/vtt")]
            | Annotated[str, bentoml.validators.ContentType("text/event-stream")]
    ):
        transcription_request = BatchTranscriptionRequest(file=file,
                                                          model=model,
                                                          language=language,
                                                          prompt=prompt,
                                                          response_format=response_format,
                                                          temperature=temperature,
                                                          timestamp_granularities=timestamp_granularities)
        response = await self.batch.batch_transcribe([transcription_request])
        return response[0]
