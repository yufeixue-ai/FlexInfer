export CUDA_VISIBLE_DEVICES=0

# eager
python benchmark/profile_benchmark.py \
    --model-name Qwen3-14B-AWQ \
    --num-seqs 8 \
    --input-len 64 \
    --output-len 128 \
    --wait 0 \
    --warmup 1 \
    --active 4 \
    --repeat 1 \
    --record-shapes \
    --profile-memory \
    --with-stack

# cuda graph
python benchmark/profile_benchmark.py \
    --model-name Qwen3-14B-AWQ \
    --num-seqs 8 \
    --input-len 64 \
    --output-len 128 \
    --capture-graphs \
    --wait 0 \
    --warmup 1 \
    --active 4