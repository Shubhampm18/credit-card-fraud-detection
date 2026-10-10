# Credit Card Fraud Detection

A machine learning project for credit card fraud detection with data drift monitoring, prediction monitoring, latency measurement, and a REST API built using FastAPI and Docker.

## Features

- Credit card fraud classification using Random Forest
- Data drift detection using Kolmogorov–Smirnov (KS) tests and Population Stability Index (PSI)
- Prediction drift monitoring
- Inference latency and throughput measurement
- FastAPI endpoints for fraud prediction
- Docker containerization
- Request logging and health checks

## Tech Stack

- Python
- Pandas, NumPy, Scikit-learn
- FastAPI and Uvicorn
- Docker
- Matplotlib
- SciPy

## Project Structure

```text
credit-card-fraud-detection/
├── data/
│   └── creditcard_sample_5000.csv
├── models/
│   └── random_forest.joblib
├── results/
│   └── baseline/
├── src/
│   ├── common.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── monitor.py
│   ├── drift_detection.py
│   ├── api.py
│   └── api_load_test.py
├── Dockerfile
├── compose.yaml
├── requirements.txt
└── README.md
```

## Dataset

The project uses the Credit Card Fraud Detection dataset from Kaggle.

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download `creditcard.csv` and place it in the project root before preparing the sample or retraining the model.

The original dataset is not included in this repository because of its size. A smaller sample is included for the existing model and monitoring workflow.

## Installation

Clone the repository:

```bash
git clone https://github.com/Shubhampm18/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train the Model

After placing `creditcard.csv` in the project root, run:

```bash
python src/prepare_data.py
python src/train.py
```

Create the reference baseline:

```bash
python src/monitor.py baseline
```

## Model Monitoring

Run the monitoring scenarios:

```bash
python src/monitor.py run --scenario control
python src/monitor.py run --scenario amount_shift
python src/monitor.py run --scenario v14_shift
python src/monitor.py run --scenario mixed
```

Run the latency benchmark:

```bash
python src/monitor.py benchmark --n 1000 --warmup 50
```

The monitoring pipeline records prediction latency, throughput, data drift, and prediction drift. Reports and plots are generated locally.

## Run the API

Start the FastAPI application:

```bash
python src/api.py
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

Available endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check |
| `/ready` | GET | Model readiness |
| `/predict` | POST | Fraud prediction |
| `/drift/status` | GET | Drift monitoring status |

## Run with Docker

Build the Docker image:

```bash
docker build -t fraud-api:local .
```

Start the container:

```bash
docker run -d --name fraud-api -p 8001:8000 -v "${PWD}/results/monitoring:/app/results/monitoring" fraud-api:local
```

Check the service:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
Invoke-RestMethod http://127.0.0.1:8001/ready
```

The monitoring directory is mounted so that request logs remain available on the host after the container stops.

To stop the container:

```bash
docker stop fraud-api
docker rm fraud-api
```

## Results

The model was evaluated using a stratified train-test split. On the current 5,000-row sample, the recorded results at the selected threshold of 0.21 were:

| Metric | Result |
|---|---:|
| Precision | 0.500 |
| Recall | 0.333 |
| F1-score | 0.400 |
| Average Precision | 0.806 |
| ROC-AUC | 0.9996 |

The monitoring experiments detected the injected data shifts and identified prediction drift in the relevant scenarios.

These results are based on a small sample containing only nine fraud cases. The evaluation metrics therefore have substantial uncertainty and should not be interpreted as evidence of production-ready fraud detection.

