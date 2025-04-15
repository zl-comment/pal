#!/bin/bash
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512


for behavior in {0..50}; do
    python -u main.py \
    --config="./configs/gpp.py" \
    --config.add_space=True \
    --config.batch_size=160 \
    --config.num_steps=500 \
    --config.log_freq=1 \
    -- \
    --scenario "AdvBenchAll" --behaviors $behavior --system_message "llama_default" \
    --model llama-2@~/autodl-tmp/pal/data/models/Llama-2-7b-chat-hf --verbose

        
    # 释放显存
    python -c "import torch; torch.cuda.empty_cache()"
done

echo "Finished."
