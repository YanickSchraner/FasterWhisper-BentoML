<div align="center">
    <h1 align="center">Serving FasterWhisper with BentoML</h1>
</div>

[FasterWhisper](https://github.com/SYSTRAN/faster-whisper) provides fast automatic speech recognition with word-level timestamps.


## Prerequisites

If you want to test the project locally, install FFmpeg on your system.

## Install dependencies

```bash
git clone https://github.com/FD-ITBS-DMS/whisper-bentoml
cd whisper-bentoml

# Recommend Python 3.12
pip install -r requirements.txt
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

2024-01-18T09:01:15+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:FasterWhisper" listening on http://localhost:3000 (Press CTRL+C to quit)

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

CURL

```bash
curl -s \
     -X POST \
     -F 'audio_file=@female.wav' \
     http://localhost:3000/transcribe
```

Python client

```python
from pathlib import Path
import bentoml

with bentoml.SyncHTTPClient('http://localhost:3000') as client:
    audio_url = 'https://example.org/female.wav'
    response = client.transcribe(file=audio_url)
    print(response)
```

## Deploy

For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/guides/containerization.html).


## Observability

BentoML automatically collects a set of default metrics for each Service and exposes them via '/metrics' endpoint.

### Prometheus (local)

To run a prometheus server locally, you need to do the following:
- Install prometheus
- Start prometheus server
```bash
prometheus --config.file=/path/to/the/file/prometheus.yml
```
- Access the web UI by visiting `http://localhost:9090`

### Grafana

Tutorial link: [link](https://docs.bentoml.com/en/latest/build-with-bentoml/observability/metrics.html#create-a-grafana-dashboard)
- Install grafana
- Change http_port to a free port like 4000 in file grafana.ini.
- Restart grafana server

### Local Development

To debug through the FasterWhisper service, you can run the service with the following script: 
```bash
python launch.py
```


