docker run --gpus 1 --rm -p 8000-8002:8000-8002    -v /home/nirbaanm/workspace/llm/models:/models    nvcr.io/nvidia/tritonserver:25.04-pyt-python-py3    tritonserver --model-repository=/models

