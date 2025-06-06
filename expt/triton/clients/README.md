# Triton Server Hello World Model

This is a simple "Hello World" model repository for NVIDIA Triton Inference Server. It contains a basic model that adds two tensors together.

## Directory Structure

```
/models/triton/
├── add_model/              # Model folder
│   ├── 1/                  # Model version folder
│   │   └── model.py        # Python model implementation
│   └── config.pbtxt        # Model configuration file
└── client.py               # Client script to test the model
```

## Model Description

The `add_model` is a simple Python model that:
- Takes two input tensors (`INPUT0` and `INPUT1`), each with shape [16] and data type FP32
- Adds the two tensors together
- Returns the result as an output tensor (`OUTPUT0`) with the same shape and data type

## Running the Server

Start the Triton server with:

```bash
docker run --gpus 1 --rm -p 8000-8002:8000-8002 \
    -v /home/nirbaanm/workspace/llm/models/triton:/models \
    nvcr.io/nvidia/tritonserver:25.04-pyt-python-py3 \
    tritonserver --model-repository=/models
```

## Testing the Model

Install the Triton client library if you haven't already:

```bash
pip install tritonclient[all]
```

Then run the client script:

```bash
python client.py
```

By default, the client uses HTTP protocol. To use gRPC instead:

```bash
python client.py --protocol grpc
```

## Next Steps

After understanding this basic model, you can:
1. Create more complex models with different backends (TensorRT, ONNX, etc.)
2. Experiment with different input/output configurations
3. Implement ensemble models that chain multiple models together
4. Try dynamic batching and other performance optimizations
