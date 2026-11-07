# API server image. Build context is the core/ directory:
#   docker build -t lumina-api -f core/Dockerfile core
# Run with a trained checkpoint mounted read-only:
#   docker run --rm -p 8000:8000 -v "$PWD/checkpoints:/models:ro" \
#     -e LUMINA_CHECKPOINT=/models/final.pt lumina-api
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LUMINA_HOST=0.0.0.0 \
    LUMINA_PORT=8000 \
    LUMINA_DEVICE=cpu

WORKDIR /app

# CPU-only torch keeps the image small; headless OpenCV avoids GUI system libraries.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch>=2.2" \
 && pip install "numpy>=1.26" "opencv-python-headless>=4.8" "scikit-image>=0.22" \
      "pillow>=10" "tqdm>=4.66" "pyyaml>=6" \
      "fastapi>=0.115" "uvicorn>=0.30" "python-multipart>=0.0.9"

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-deps .

RUN useradd --system --uid 10001 --no-create-home lumina
USER lumina

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "nta_zerodce.api"]
