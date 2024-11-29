import time
from pathlib import Path

import bentoml


class TestIntegration:

    def test_transcribe_task_endpoint(self):
        # given
        file = Path(__file__).resolve().parent.parent / "assets" / "example_audio.mp3"
        client = bentoml.SyncHTTPClient('http://localhost:8002')

        # when
        task = client.task_transcribe.submit(file=file)
        while True:
            status = task.get_status()
            if str(status) == "ResultStatus.SUCCESS":
                break
            time.sleep(5)
        result = task.get()

        # then
        assert "I am just a sample audio text." in result

    def test_transcribe_streaming_endpoint(self):
        # given
        client = bentoml.SyncHTTPClient("http://localhost:8002")
        file = Path(__file__).resolve().parent.parent / "assets" / "example_audio.mp3"
        data_chunks = []

        # when
        for data_chunk in client.streaming_transcribe(file=file):
            data_chunks.append(data_chunk)
        client.close()

        # then
        assert data_chunks is not None
