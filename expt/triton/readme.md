# GPU
docker run -d --gpus 1 --rm -p 8000-8002:8000-8002    -v /home/nirbaanm/workspace/git/llm/expt/triton/models:/models    nvcr.io/nvidia/tritonserver:25.04-pyt-python-py3    tritonserver --model-repository=/models

# CPU
$env:model_local_base="D:/ds/work/workspace/git/llm/expt/triton/models"
docker run --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti -v D:/ds/work/workspace/git/llm/expt/triton/models:/models  nvcr.io/nvidia/tritonserver:25.04-pyt-python-py3  tritonserver --model-repository=/models

# Notes
## Linear Regression Examples
Two linear regression models have been implemented for demonstration:

### 1. Simple Linear Regression Model
- Located at `models/linear_regression_model/`
- Implements a basic linear function y = 2x + 5
- Hardcoded parameters (weights=2.0, bias=5.0)
- Client script: `clients/client_linear_regression.py`

### 2. Scikit-learn Linear Regression Model
- Located at `models/sklearn_regression/`
- Uses scikit-learn's LinearRegression algorithm
- Trains on synthetic data during initialization (y = 3x + 2 with noise)
- Saves and loads the model using joblib
- Client script: `clients/client_sklearn_regression.py`

## Running the Clients

After starting the Triton server with the commands above, run either client:

```bash
# Simple Linear Regression
python clients/client_linear_regression.py --visualize

# Scikit-learn Regression
python clients/client_sklearn_regression.py --visualize
```

## Dependencies

Both models require the following dependencies:
- NumPy
- tritonclient[all] (for the clients)
- matplotlib (for visualization)
- scikit-learn and joblib (only for the sklearn_regression model)

Install them using pip:

```bash
pip install numpy tritonclient[all] matplotlib scikit-learn joblib
```

## Command Line Options

Both client scripts support:
- `--protocol`: Choose between 'http' (default) or 'grpc'
- `--visualize`: Enable result visualization with matplotlib
- `--num-points`: Number of test data points to generate

## Monitoring with Prometheus and Grafana

The setup includes a complete monitoring stack using Prometheus and Grafana to visualize Triton Inference Server metrics.

### Setup

1. Ensure the Triton server is started with metrics enabled:
   ```powershell
   docker run --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8080:8080 --ulimit stack=67108864 -v D:/ds/work/workspace/git/llm/expt/triton/models:/models nvcr.io/nvidia/tritonserver:25.04-pyt-python-py3 tritonserver --model-repository=/models --metrics-port=8080
   ```

2. Alternatively, use the provided Docker Compose file to start the entire stack (Triton, Prometheus, and Grafana):
   ```powershell
   docker-compose up -d
   ```

3. Generate Load with the Load Test Script Use the load test script I created to send requests to your models:

python clients/load_test.py --model linear_regression_model --threads 4 --duration 300 --qps 20

This will:
Send requests to the linear_regression_model
Run 4 concurrent threads
Run for 300 seconds (5 minutes)
Target 20 queries per second per thread (80 QPS total)


### Accessing Dashboards

- **Prometheus**: Available at http://localhost:9090
- **Grafana**: Available at http://localhost:3000 (login with admin/admin)

### Available Metrics

Triton exposes several metrics including:

- Request success/failure counts per model
- Queue and compute durations
- Inference request latencies
- Memory usage statistics

The pre-configured dashboard in Grafana shows:

- Successful inference requests
- Failed inference requests
- Queue and compute durations
- Performance metrics

