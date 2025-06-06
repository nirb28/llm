# GPU
docker run -d --gpus 1 --rm -p 8000-8002:8000-8002    -v /home/nirbaanm/workspace/git/llm/expt/triton/models:/models    nvcr.io/nvidia/tritonserver:25.04-pyt-python-py3    tritonserver --model-repository=/models

# CPU
$env:model_local_base="D:/ds/sync/gdrive_ds/My Drive/work/gdrive-workspaces/git/llm/expt/triton/models"
docker run --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti -v "/mnt/d/ds/sync/gdrive_ds/My Drive/work/gdrive-workspaces/git/llm/expt/triton/models":/models  nvcr.io/nvidia/tritonserver:25.04-pyt-python-py3  tritonserver --model-repository=/models