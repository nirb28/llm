export trt_base=/home/nirbaanm/workspace/llm/tools/TensorRT-LLM
cd $trt_base/examples/models/core/llama
export DRAFT_MODEL_PATH=/home/nirbaanm/workspace/llm/models/hf/meta-llama/Llama-3.1-8B-Instruct/
export TARGET_MODEL_PATH=/home/nirbaanm/workspace/llm/models/hf/meta-llama/Llama-3.3-70B-Instruct/

export expt_workarea=/workspace/llm/models/expt/sd
export DRAFT_CKPT_PATH=$expt_workarea/ckpt-draft
export TARGET_CKPT_PATH=$expt_workarea/ckpt-target
export DRAFT_ENGINE_PATH=$expt_workarea/engine-draft
export TARGET_ENGINE_PATH=$expt_workarea/engine-target
export MAX_BATCH_SIZE=4
export MAX_DRAFT_LEN=10
export MAX_INPUT_LEN=3200
export MAX_SEQ_LEN=4800

python3 convert_checkpoint.py \
    --model_dir=${DRAFT_MODEL_PATH} \
    --output_dir=${DRAFT_CKPT_PATH} \
    --dtype=float16

python3 convert_checkpoint.py \
    --model_dir=${TARGET_MODEL_PATH} \
    --output_dir=${TARGET_CKPT_PATH} \
    --dtype=float16

trtllm-build \
    --checkpoint_dir=${DRAFT_CKPT_PATH} \
    --output_dir=${DRAFT_ENGINE_PATH} \
    --gemm_plugin=float16 \
    --use_paged_context_fmha=enable \
    --max_batch_size=${MAX_BATCH_SIZE} \
    --max_input_len=${MAX_INPUT_LEN} \
    --max_seq_len=${MAX_SEQ_LEN}

trtllm-build \
    --checkpoint_dir=${TARGET_CKPT_PATH} \
    --output_dir=${TARGET_ENGINE_PATH} \
    --gemm_plugin=float16 \
    --use_paged_context_fmha=enable \
    --speculative_decoding_mode=draft_tokens_external \
    --max_batch_size=${MAX_BATCH_SIZE} \
    --max_draft_len=${MAX_DRAFT_LEN} \
    --max_input_len=${MAX_INPUT_LEN} \
    --max_seq_len=${MAX_SEQ_LEN}