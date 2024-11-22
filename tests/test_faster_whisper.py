import pytest

from pathlib import Path

from service import FasterWhisper
from utils import ResponseFormat


class TestFasterWhisper:

    def test_transcribe(self):
        # given
        faster_whisper_service = FasterWhisper()
        file = Path("assets/example_audio.mp3")

        # when
        transcription = faster_whisper_service.transcribe(file, response_format=ResponseFormat.JSON)

        # then
        assert transcription is not None

    def test_transcribe_temperature(self):
        # given
        faster_whisper_service = FasterWhisper()
        file = Path("assets/example_audio.mp3")
        temperature = 0.4

        # when
        transcription = faster_whisper_service.transcribe(file,
                                                          temperature=temperature,
                                                          response_format=ResponseFormat.JSON)

        # then
        assert transcription is not None

    @pytest.mark.parametrize("response_format, timestamp_granularities", [
        (ResponseFormat.JSON, []),
        (ResponseFormat.VERBOSE_JSON, ["word"]),
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
