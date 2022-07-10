#!/bin/bash

set -eu

TRAIN=false
DECODE=false
EVAL=false
SPLIT=test
EPOCHS=10
MODEL="facebook/bart-base"
ACC_GRAD=4
BATCH_SIZE=8
MAX_THREADS=8
GPUS=1
SEED=42
BLEU_REF=""
OUT_FILENAME="test.out"
VAL_CHECK_INTERVAL=1.0

while [ $# -gt 0 ]; do
    case "$1" in
        "-t"|"--train") TRAIN=true ;;
        "-d"|"--decode") DECODE=true ;;
        "-e"|"--eval") EVAL=true ;;
        "--beam_size") BEAM_SIZE="$2"; shift ;;
        "--bleu_ref") BLEU_REF="$2"; shift ;;
        "--decode_in_dir") DECODE_IN_DIR="$2"; shift ;;
        "--epochs") EPOCHS="$2"; shift ;;
        "--eval_ref") EVAL_REF="$2"; shift ;;
        "--experiment") EXPERIMENT="$2"; shift ;;
        "--model") MODEL="$2"; shift ;;
        "--out_filename") OUT_FILENAME="$2"; shift ;;
        "--split") SPLIT="$2"; shift ;;
        "--seed") SEED="$2"; shift ;;
        "--train_in_dir") TRAIN_IN_DIR="$2"; shift ;;
        "--val_check_interval") VAL_CHECK_INTERVAL="$2"; shift ;;
        *) echo "Unknown option: $1" ;;
    esac
    shift
done

if $TRAIN; then
    ./train.py \
        --in_dir "$TRAIN_IN_DIR" \
        --experiment "$EXPERIMENT" \
        --gpus "$GPUS" \
        --model_name "$MODEL" \
        --batch_size "$BATCH_SIZE" \
        --accumulate_grad_batches "$ACC_GRAD" \
        --max_epochs "$EPOCHS"  \
        --bleu_ref "$BLEU_REF" \
        --val_check_interval "$VAL_CHECK_INTERVAL" \
        --seed "$SEED"
fi

if $DECODE; then
    ./decode.py \
        --experiment "$EXPERIMENT" \
        --in_dir "$DECODE_IN_DIR" \
        --split "test" \
        --gpus "$GPUS" \
        --out_filename "$OUT_FILENAME" \
        --beam_size "$BEAM_SIZE" \
        --seed "$SEED"
fi

if $EVAL; then
    HYP_FILE="experiments/${EXPERIMENT}/$OUT_FILENAME"
    EVAL_FILE="experiments/${EXPERIMENT}/${OUT_FILENAME}.eval.json"
    gem_metrics \
        "$HYP_FILE"\
        -r "$EVAL_REF" \
        --metric-list msttr ngrams bleu meteor nubia bleurt \
        > "$EVAL_FILE"

    printf "[RES] "
    for METRIC in "bleu" \
         "meteor" \
         "bleurt" \
         "nubia_semsim" \
         "nubia_contr" \
         "nubia_neutr" \
         "nubia_agree" \
         "nubia" \
         "unique-1" \
         "cond_entropy-2" \
         "msttr-100" \
         "nubia_perpl" \
         "mean_pred_length"; do
        RES=$(jq ".\"${METRIC}\"" "$EVAL_FILE")

        if [ $METRIC = "meteor" ]; then 
            RES=$(echo "$RES * 100" | bc)
        fi
        printf "%.2f;" "$RES"
    done
    echo
fi