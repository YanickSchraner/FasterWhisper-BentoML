import logging
from typing import TYPE_CHECKING, Annotated, AsyncGenerator, List, Optional

import bentoml
import huggingface_hub
from bentoml.validators import ContentType
from fastapi import FastAPI

from api_models.enums import ResponseFormat, Task
from api_models.input_models import ModelName, ValidatedLanguage, ValidatedResponseFormat, ValidatedTemperature, \
    TimestampGranularities, BatchTranscriptionRequest, validate_timestamp_granularities, hf_model_info_to_model_object
from api_models.output_models import segments_to_streaming_response, ModelListResponse, segments_to_response, \
    ModelObject
from config import WhisperModelConfig, faster_whisper_config
from core import Segment
from logger import configure_logging
from model_manager import WhisperModelManager

if TYPE_CHECKING:
    from huggingface_hub.hf_api import ModelInfo

from pathlib import Path
from pydantic import Field

from bentoml.exceptions import NotFound

logger = logging.getLogger("bentoml")

fastapi = FastAPI()

configure_logging()


@bentoml.service(
    traffic={"timeout": 3000},
    resources={
        "gpu": 1,
        "memory": "8Gi",
    },
)
@bentoml.mount_asgi_app(fastapi, path="/v1")
class FasterWhisper:
    def __init__(self):
        self.model_manager = WhisperModelManager(WhisperModelConfig())

    @bentoml.api(route="/v1/audio/transcriptions")
    def transcribe(
            self,
            file: Annotated[Path, ContentType("audio/mpeg")],
            model: Optional[ModelName] = Field(
                default=faster_whisper_config.default_model_name,
                description="Whisper model to load"
            ),
            language: Optional[ValidatedLanguage] = Field(
                default=faster_whisper_config.default_language,
                description='The language spoken in the audio. It should be a language code such as "en" or "fr". If '
                            'not set, the language will be detected in the first 30 seconds of audio.'
            ),
            prompt: Optional[str] = Field(
                default=faster_whisper_config.default_prompt,
                description="Optional text string or iterable of token ids to provide as a prompt for the first window.",
            ),
            response_format: Optional[ValidatedResponseFormat] = Field(
                default=faster_whisper_config.default_response_format,
                description="The format of the output, in one of these options: `json`, `text`, `srt`, `verbose_json`, "
                            "or `vtt`.",
            ),
            temperature: Optional[ValidatedTemperature] = Field(
                default=faster_whisper_config.default_temperature,
                description="Temperature value, which can either be a single float or a list of floats.",
            ),
            timestamp_granularities: Optional[TimestampGranularities] = Field(
                default=faster_whisper_config.default_timestamp_granularities,
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
        logger.debug(f"number of requests processed: {len(requests)}")
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
                default=faster_whisper_config.default_model_name,
                description="Whisper model to load"
            ),
            language: Optional[ValidatedLanguage] = Field(
                default=faster_whisper_config.default_language,
                description='The language spoken in the audio. It should be a language code such as "en" or "fr". If '
                            'not set, the language will be detected in the first 30 seconds of audio.'
            ),
            prompt: Optional[str] = Field(
                default=faster_whisper_config.default_prompt,
                description="Optional text string or iterable of token ids to provide as a prompt for the first window.",
            ),
            response_format: Optional[ValidatedResponseFormat] = Field(
                default=faster_whisper_config.default_response_format,
                description="The format of the output, in one of these options: `json`, `text`, `srt`, `verbose_json`, "
                            "or `vtt`.",
            ),
            temperature: Optional[ValidatedTemperature] = Field(
                default=faster_whisper_config.default_temperature,
                description="Temperature value, which can either be a single float or a list of floats.",
            ),
            timestamp_granularities: Optional[TimestampGranularities] = Field(
                default=faster_whisper_config.default_timestamp_granularities,
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
                default=faster_whisper_config.default_model_name,
                description="Whisper model to load"
            ),
            language: Optional[ValidatedLanguage] = Field(
                default=faster_whisper_config.default_language,
                description='The language spoken in the audio. It should be a language code such as "en" or "fr". If '
                            'not set, the language will be detected in the first 30 seconds of audio.'
            ),
            prompt: Optional[str] = Field(
                default=faster_whisper_config.default_prompt,
                description="Optional text string or iterable of token ids to provide as a prompt for the first window.",
            ),
            response_format: Optional[ValidatedResponseFormat] = Field(
                default=faster_whisper_config.default_response_format,
                description="The format of the output, in one of these options: `json`, `text`, `srt`, `verbose_json`, "
                            "or `vtt`.",
            ),
            temperature: Optional[ValidatedTemperature] = Field(
                default=faster_whisper_config.default_temperature,
                description="Temperature value, which can either be a single float or a list of floats.",
            ),
            timestamp_granularities: Optional[TimestampGranularities] = Field(
                default=faster_whisper_config.default_timestamp_granularities,
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

    @bentoml.api(route="/v1/audio/translations")
    def translate(
            self,
            file: Annotated[Path, ContentType("audio/mpeg")],
            model: Optional[ModelName] = Field(
                default="large-v3", description="Whisper model to load"
            ),
            prompt: Optional[str] = Field(
                default=faster_whisper_config.default_prompt,
                description="An optional text to guide the model's style or continue a previous audio segment. The "
                            "prompt should be in English.",
            ),
            response_format: Optional[ValidatedResponseFormat] = Field(
                default=faster_whisper_config.default_response_format,
                description=f"One of: {[format for format in ResponseFormat]}",
            ),
            temperature: Optional[ValidatedTemperature] = Field(
                default=faster_whisper_config.default_temperature,
                description="Temperature value, which can either be a single float or a list of floats.",
            )
    ) -> (
            Annotated[str, bentoml.validators.ContentType("text/plain")]
            | Annotated[str, bentoml.validators.ContentType("application/json")]
            | Annotated[str, bentoml.validators.ContentType("text/vtt")]
            | Annotated[str, bentoml.validators.ContentType("text/event-stream")]
    ):
        return self._translate_audio(file, model, prompt, response_format, temperature)

    @fastapi.get("/models")
    def get_models(self) -> ModelListResponse:
        models = huggingface_hub.list_models(
            library="ctranslate2", tags="automatic-speech-recognition", cardData=True
        )
        models = list(models)
        models.sort(key=lambda model: model.downloads, reverse=True)
        transformed_models = [hf_model_info_to_model_object(model) for model in models]
        return ModelListResponse(data=transformed_models)

    @fastapi.get("/models/{model_name:path}")
    def get_model(self,
                  model_name=Annotated[str, Path(example="Systran/faster-distil-whisper-large-v3")]) -> ModelObject:
        models = huggingface_hub.list_models(
            model_name=model_name,
            library="ctranslate2",
            tags="automatic-speech-recognition",
            cardData=True,
        )
        models = list(models)
        models.sort(key=lambda model: model.downloads, reverse=True)  # type: ignore  # noqa: PGH003
        if len(models) == 0:
            raise NotFound(message="No models found.")
        exact_match: ModelInfo | None = None
        for model in models:
            if model.id == model_name:
                exact_match = model
                break
        if exact_match is None:
            raise NotFound(
                message=f"Model doesn't exists. Possible matches: {', '.join([model.id for model in models])}",
            )
        return hf_model_info_to_model_object(exact_match)

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

    def _translate_audio(self, file, model, prompt, response_format, temperature):
        with self.model_manager.load_model(model) as whisper:
            segments, transcription_info = whisper.transcribe(
                file,
                task=Task.TRANSLATE,
                initial_prompt=prompt,
                temperature=temperature,
                word_timestamps=response_format == ResponseFormat.VERBOSE_JSON,
            )
        segments = Segment.from_faster_whisper_segments(segments)
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
