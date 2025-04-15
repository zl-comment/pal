#!/bin/bash
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

ATTACK="gcg"


start=${1:-0}
end=${2:-50}

for behavior in $(seq $start $end); do
    python -u main.py \
        --config="./configs/${ATTACK}.py" \
        --config.add_space=True \
        --config.batch_size=8 \
        --config.mini_batch_size=8 \
        --config.num_steps=2000 \
        --config.log_freq=1 \
        --config.fixed_params=True \
        -- \
        --scenario "AdvBenchAll" --behaviors $behavior --system_message "guanaco" \
        --model 'llama-2@/home/zl/ZLCODE/model/guanaco-7B-HF' --verbose
        
    # 释放显存
    python -c "import torch; torch.cuda.empty_cache()"
done

echo "Finished."
