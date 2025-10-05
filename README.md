# EcoSentinel — AI Environmental Change Detector

EcoSentinel detects change between BEFORE and AFTER satellite images. It builds a change map (SSIM + OpenCV), reports SSIM and estimated % area changed, and generates a short plain-English summary using Meta Llama 3.1 served via the Cerebras Inference API. The app is containerized with Docker for a clean, reproducible run.

## Quick start (Docker)
1) Create `.env` in the project root:
CEREBRAS_API_KEY=cbsk_your_key_here
CEREBRAS_MODEL=llama3.1-8b

2) Build and run:
```bash
docker build -t ecosentinel .
docker run -d --name ecosentinel_run -p 8530:8501 --env-file .env ecosentinel
Open: http://localhost:8530

3) How to Use it :

Turn OFF “Use built-in demo pair”.

Upload BEFORE and AFTER images of the same area (similar scale).

Click “Generate AI Environmental Report” to get the summary and recommendations.

4) Stack :

Streamlit (UI) · OpenCV + SSIM (change detection) · Meta Llama 3.1 via Cerebras (report) · Docker (packaging)

5) Troubleshooting :

401 Unauthorized when generating report: check/refresh CEREBRAS_API_KEY and ensure .env is loaded.

Port already in use: change mapping, e.g. -p 8531:8501 and open http://localhost:8531.

Model not found (404): use llama3.1-8b (default).


