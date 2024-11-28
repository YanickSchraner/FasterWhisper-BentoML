import json
import pytest

from pathlib import Path

from service import FasterWhisper
from api_models import ResponseFormat, TimestampGranularity
from bentoml.exceptions import InvalidArgument


class TestFasterWhisper:

    def test_transcribe_standard_case(self):
        # given
        faster_whisper_service = FasterWhisper()
        file = Path("assets/example_audio.mp3")

        # when
        transcription = faster_whisper_service.transcribe(file, response_format=ResponseFormat.JSON)

        # then
        assert transcription is not None

    @pytest.mark.parametrize("temperature", [
        [0.3, 0.6],
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        0.0
    ])
    def test_transcribe_temperature(self, temperature):
        # given
        faster_whisper_service = FasterWhisper()
        file = Path("assets/example_audio.mp3")

        # when
        transcription = faster_whisper_service.transcribe(file,
                                                          temperature=temperature,
                                                          response_format=ResponseFormat.JSON)

        # then
        assert transcription is not None

    @pytest.mark.parametrize("response_format, timestamp_granularities", [
        (ResponseFormat.JSON, []),
        (ResponseFormat.VERBOSE_JSON, [TimestampGranularity.WORD]),
        (ResponseFormat.SRT, []),
        (ResponseFormat.TEXT, []),
        (ResponseFormat.VTT, [])
    ])
    def test_transcribe_response_format(self, response_format, timestamp_granularities):
        # given
        faster_whisper_service = FasterWhisper()
        file = Path("assets/example_audio.mp3")

        # when
        transcription = faster_whisper_service.transcribe(file,
                                                          response_format=response_format,
                                                          timestamp_granularities=timestamp_granularities)

        # then
        assert transcription is not None

    def test_response_format_verbose_timestamp_granularities_segment(self):
        # given
        faster_whisper_service = FasterWhisper()
        file = Path("assets/example_audio.mp3")
        response_format = ResponseFormat.VERBOSE_JSON
        timestamp_granularities = [TimestampGranularity.SEGMENT]

        # when/then
        with pytest.raises(InvalidArgument):
            faster_whisper_service.transcribe(file,
                                              response_format=response_format,
                                              timestamp_granularities=timestamp_granularities)

    def test_response_format_verbose_timestamp_granularities_word(self):
        # given
        faster_whisper_service = FasterWhisper()
        file = Path("assets/example_audio.mp3")
        response_format = ResponseFormat.VERBOSE_JSON
        timestamp_granularities = [TimestampGranularity.WORD]

        # when
        transcription = faster_whisper_service.transcribe(file,
                                          response_format=response_format,
                                          timestamp_granularities=timestamp_granularities)

        # then
        assert json.loads(transcription)["words"] is not None

    @pytest.mark.asyncio
    async def test_transcribe_streaming(self):
        # given
        faster_whisper_service = FasterWhisper()
        file = Path("assets/example_audio.mp3")
        chunks = []

        # when
        async for chunk in faster_whisper_service.streaming_transcribe(file, response_format=ResponseFormat.JSON):
            chunks.append(chunk)

        # then
        assert chunks is not None
