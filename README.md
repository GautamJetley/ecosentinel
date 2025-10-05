# EcoSentinel — AI Environmental Change Detector

Upload BEFORE & AFTER satellite images to:
- compute a change map (SSIM + OpenCV),
- estimate % area impacted,
- generate a concise environmental report with Meta Llama (via Groq),
- optionally enhance with Cerebras (“Mitigation & Next Steps”),
- all packaged in Docker for one-command reproducibility.

## Run (Docker)
```bash
docker build -t ecosentinel .
docker run -p 8530:8501 \
  -e GROQ_API_KEY=$GROQ_API_KEY \
  -e CEREBRAS_API_KEY=$CEREBRAS_API_KEY \
  ecosentinel
# open http://localhost:8530

