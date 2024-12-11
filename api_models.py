import enum
import logging
from pathlib import Path
from typing import Iterable, Generator, Annotated, Optional, Union, List
from typing import Literal

from bentoml.exceptions import InvalidArgument
from faster_whisper.transcribe import TranscriptionInfo
from huggingface_hub import ModelInfo
from pydantic import BaseModel, BeforeValidator
from pydantic import ConfigDict, Field

from core import Segment, segments_to_text, segments_to_vtt, segments_to_srt, Transcription, Word

logger = logging.getLogger(__name__)

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


class TimestampGranularity(enum.StrEnum):
    SEGMENT = "segment"
    WORD = "word"


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


ValidatedTemperature = Annotated[
    Union[float, List[float]],
    BeforeValidator(_convert_temperature),
]


def _process_empty_response_format(response_format: Optional[str]) -> Optional[str]:
    if response_format == '':
        return DEFAULT_RESPONSE_FORMAT
    return response_format


class ResponseFormat(enum.StrEnum):
    TEXT = "text"
    JSON = "json"
    VERBOSE_JSON = "verbose_json"
    SRT = "srt"
    VTT = "vtt"


ValidatedResponseFormat = Annotated[
    ResponseFormat,
    BeforeValidator(_process_empty_response_format)
]


class Task(enum.StrEnum):
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


def _process_empty_language(language: Optional[str]) -> Optional[str]:
    if language == '':
        return DEFAULT_LANGUAGE
    return language


class Language(enum.StrEnum):
    AF = "af"
    AM = "am"
    AR = "ar"
    AS = "as"
    AZ = "az"
    BA = "ba"
    BE = "be"
    BG = "bg"
    BN = "bn"
    BO = "bo"
    BR = "br"
    BS = "bs"
    CA = "ca"
    CS = "cs"
    CY = "cy"
    DA = "da"
    DE = "de"
    EL = "el"
    EN = "en"
    ES = "es"
    ET = "et"
    EU = "eu"
    FA = "fa"
    FI = "fi"
    FO = "fo"
    FR = "fr"
    GL = "gl"
    GU = "gu"
    HA = "ha"
    HAW = "haw"
    HE = "he"
    HI = "hi"
    HR = "hr"
    HT = "ht"
    HU = "hu"
    HY = "hy"
    ID = "id"
    IS = "is"
    IT = "it"
    JA = "ja"
    JW = "jw"
    KA = "ka"
    KK = "kk"
    KM = "km"
    KN = "kn"
    KO = "ko"
    LA = "la"
    LB = "lb"
    LN = "ln"
    LO = "lo"
    LT = "lt"
    LV = "lv"
    MG = "mg"
    MI = "mi"
    MK = "mk"
    ML = "ml"
    MN = "mn"
    MR = "mr"
    MS = "ms"
    MT = "mt"
    MY = "my"
    NE = "ne"
    NL = "nl"
    NN = "nn"
    NO = "no"
    OC = "oc"
    PA = "pa"
    PL = "pl"
    PS = "ps"
    PT = "pt"
    RO = "ro"
    RU = "ru"
    SA = "sa"
    SD = "sd"
    SI = "si"
    SK = "sk"
    SL = "sl"
    SN = "sn"
    SO = "so"
    SQ = "sq"
    SR = "sr"
    SU = "su"
    SV = "sv"
    SW = "sw"
    TA = "ta"
    TE = "te"
    TG = "tg"
    TH = "th"
    TK = "tk"
    TL = "tl"
    TR = "tr"
    TT = "tt"
    UK = "uk"
    UR = "ur"
    UZ = "uz"
    VI = "vi"
    YI = "yi"
    YO = "yo"
    YUE = "yue"
    ZH = "zh"


# None needed to return None from Validator
ValidatedLanguage = Annotated[
    Language | None,
    BeforeValidator(_process_empty_language)
]


class ModelObject(BaseModel):
    id: str
    """The model identifier, which can be referenced in the API endpoints."""
    created: int
    """The Unix timestamp (in seconds) when the model was created."""
    object_: Literal["model"] = Field(serialization_alias="object")
    """The object type, which is always "model"."""
    owned_by: str
    """The organization that owns the model."""
    language: list[str] = Field(default_factory=list)
    """List of ISO 639-3 supported by the model. It's possible that the list will be empty. This field is not a part of the OpenAI API spec and is added for convenience."""  # noqa: E501

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "id": "Systran/faster-whisper-large-v3",
                    "created": 1700732060,
                    "object": "model",
                    "owned_by": "Systran",
                },
                {
                    "id": "Systran/faster-distil-whisper-large-v3",
                    "created": 1711378296,
                    "object": "model",
                    "owned_by": "Systran",
                },
                {
                    "id": "bofenghuang/whisper-large-v2-cv11-french-ct2",
                    "created": 1687968011,
                    "object": "model",
                    "owned_by": "bofenghuang",
                },
            ]
        },
    )


class ModelListResponse(BaseModel):
    data: list[ModelObject]
    object: Literal["list"] = "list"


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


def segments_to_response(
        segments: Iterable[Segment],
        transcription_info: TranscriptionInfo,
        response_format: ResponseFormat,
):
    segments = list(segments)
    if response_format == ResponseFormat.TEXT:  # noqa: RET503
        return segments_to_text(segments)
    elif response_format == ResponseFormat.JSON:
        return TranscriptionJsonResponse.from_segments(segments).model_dump_json()
    elif response_format == ResponseFormat.VERBOSE_JSON:
        return TranscriptionVerboseJsonResponse.from_segments(segments, transcription_info).model_dump_json()
    elif response_format == ResponseFormat.VTT:
        return "".join(segments_to_vtt(segment, i) for i, segment in enumerate(segments))
    elif response_format == ResponseFormat.SRT:
        return "".join(segments_to_srt(segment, i) for i, segment in enumerate(segments))


def format_as_sse(data: str) -> str:
    return f"data: {data}\n\n"


def segments_to_streaming_response(
        segments: Iterable[Segment],
        transcription_info: TranscriptionInfo,
        response_format: ResponseFormat,
):
    def segment_responses() -> Generator[str, None, None]:
        for i, segment in enumerate(segments):
            if response_format == ResponseFormat.TEXT:
                data = segment.text
            elif response_format == ResponseFormat.JSON:
                data = TranscriptionJsonResponse.from_segments([segment]).model_dump_json()
            elif response_format == ResponseFormat.VERBOSE_JSON:
                data = TranscriptionVerboseJsonResponse.from_segment(segment, transcription_info).model_dump_json()
            elif response_format == ResponseFormat.VTT:
                data = segments_to_vtt(segment, i)
            elif response_format == ResponseFormat.SRT:
                data = segments_to_srt(segment, i)
            else:
                raise ValueError(f"Unknown response format: {response_format}")
            yield format_as_sse(data)

    return segment_responses()


# https://platform.openai.com/docs/api-reference/audio/json-object
class TranscriptionJsonResponse(BaseModel):
    text: str

    @classmethod
    def from_segments(cls, segments: list[Segment]):
        return cls(text=segments_to_text(segments))

    @classmethod
    def from_transcription(cls, transcription: Transcription):
        return cls(text=transcription.text)


# https://platform.openai.com/docs/api-reference/audio/verbose-json-object
class TranscriptionVerboseJsonResponse(BaseModel):
    task: str = "transcribe"
    language: str
    duration: float
    text: str
    words: list[Word]
    segments: list[Segment]

    @classmethod
    def from_segment(cls, segment: Segment, transcription_info: TranscriptionInfo):
        return cls(
            language=transcription_info.language,
            duration=segment.end - segment.start,
            text=segment.text,
            words=(segment.words if isinstance(segment.words, list) else []),
            segments=[segment],
        )

    @classmethod
    def from_segments(
            cls, segments: list[Segment], transcription_info: TranscriptionInfo
    ):
        return cls(
            language=transcription_info.language,
            duration=transcription_info.duration,
            text=segments_to_text(segments),
            segments=segments,
            words=Word.from_segments(segments),
        )

    @classmethod
    def from_transcription(cls, transcription: Transcription):
        return cls(
            language="english",  # FIX: hardcoded
            duration=transcription.duration,
            text=transcription.text,
            words=transcription.words,
            segments=[],  # FIX: hardcoded
        )


def validate_timestamp_granularities(response_format, timestamp_granularities):
    if (
            timestamp_granularities != DEFAULT_TIMESTAMP_GRANULARITIES
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
    model: Optional[str]
    language: Optional[str]
    prompt: Optional[str]
    response_format: Optional[ResponseFormat]
    temperature: Optional[Union[float, List[float]]]
    timestamp_granularities: Optional[TimestampGranularities]


DEFAULT_RESPONSE_FORMAT = ResponseFormat.JSON
DEFAULT_TEMPERATURE = 0.0
DEFAULT_LANGUAGE = None
DEFAULT_TRANSCRIPTION_MODEL = "large-v3"
DEFAULT_PROMPT = None
DEFAULT_TIMESTAMP_GRANULARITIES = [TimestampGranularity.SEGMENT]
