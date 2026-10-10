# Credit Card Fraud Detection API - local Docker image
#
# Runs the FastAPI service (src/api.py) with the *pre-trained* model. The raw
# dataset and training code are NOT copied in: the API only needs the model
# artifact, the feature schema and the saved baseline reference.
#
# Build (from the project root):
#   docker build -t fraud-api:local .
# Run:
#   docker run --rm -p 8000:8000 -v "%CD%\results\monitoring:/app/results/monitoring" fraud-api:local

FROM python:3.11-slim

# Fail fast, keep stdout unbuffered so container logs are readable.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Only what inference needs: source, the trained model, the baseline reference
# used for drift comparison, and the tiny sample used by the load test.
# The 150 MB creditcard.csv is deliberately excluded (see .dockerignore).
COPY src/ ./src/
COPY models/ ./models/
COPY results/baseline/ ./results/baseline/
COPY data/creditcard_sample_5000.csv ./data/creditcard_sample_5000.csv

# Monitoring output directory (normally bind-mounted from the host so records
# survive container removal). Created here so the app can also run unmounted.
RUN mkdir -p /app/results/monitoring/runs /app/results/plots

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV FRAUD_PORT=8000 \
    FRAUD_HOST=0.0.0.0 \
    FRAUD_MONITORING_DIR=/app/results/monitoring \
    FRAUD_MODEL_PATH=/app/models/random_forest.joblib \
    FRAUD_MAX_BATCH_SIZE=1000 \
    FRAUD_DRIFT_WINDOW_SIZE=500

EXPOSE 8000

# Health check hits the cheap liveness endpoint (Python stdlib, no curl needed).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys,os; \
url='http://127.0.0.1:'+os.environ.get('FRAUD_PORT','8000')+'/health'; \
sys.exit(0 if urllib.request.urlopen(url, timeout=3).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]