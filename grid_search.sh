#!/bin/bash
cd /root/autodl-tmp/mmtok_project/MMTok
conda activate mmtok
# export HF_HUB_OFFLINE=1

BASE_MODEL="/root/autodl-tmp/hf_cache/hub/models--liuhaotian--llava-v1.6-vicuna-7b/snapshots/deae57a8c0ccb0da4c2661cc1891cc9d06503d11"
TASK="textvqa_val"
BATCH_SIZE=1
LIMIT="--limit 1000"   # 可改为 "" 跑全量 5000

TOKENS_LIST=(16 32 64)
ALPHA_LIST=(0.3 0.5 0.7)

SUMMARY_FILE="./grid_search_summary.csv"
echo "tokens,alpha,exact_match" > $SUMMARY_FILE

for tokens in "${TOKENS_LIST[@]}"; do
    for alpha in "${ALPHA_LIST[@]}"; do
        echo "Running: tokens=$tokens, alpha=$alpha"
        OUTPUT_DIR="./results_grid/tokens_${tokens}_alpha_${alpha}"
        python -m lmms_eval \
            --model llava_mmtok \
            --model_args pretrained="$BASE_MODEL",target_vision_tokens=$tokens,alpha=$alpha,softmax_tv_temperature=0.02,softmax_vv_temperature=0.2,conv_template="vicuna_v1" \
            --tasks $TASK \
            $LIMIT \
            --batch_size $BATCH_SIZE \
            --output_path $OUTPUT_DIR \
            --log_samples
        # 提取准确率
        ACC=$(python -c "import json, glob; f=glob.glob('$OUTPUT_DIR/snapshots__*/results.json')[0]; print(json.load(open(f))['results']['textvqa_val']['exact_match,none'])" 2>/dev/null || echo "ERROR")
        echo "$tokens,$alpha,$ACC" >> $SUMMARY_FILE
    done
done

echo "Grid search finished. Summary:"
cat $SUMMARY_FILE