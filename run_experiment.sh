#!/bin/bash

BEAM_SIZE=1
SEED=42
SPLIT=test
TRAIN=""
DECODE=""
EVAL=""
EXPERIMENT=""

while [ $# -gt 0 ]; do
    case "$1" in
        "-t"|"--train") TRAIN="-t" ;;
        "-d"|"--decode") DECODE="-d" ;;
        "-e"|"--eval") EVAL="-e" ;;
        "-b"|"--beam_size") BEAM_SIZE="$2"; shift ;;
        "-s"|"--seed") SEED="$2"; shift ;;
        "-f"|"--fewshot") FEWSHOT="$2"; shift ;;
        *) EXPERIMENT="$1" ;;
    esac
    shift
done

case "$EXPERIMENT" in
    "copy")
        if [ -n $DECODE ]; then
            mkdir -p "experiments/copy/"
            cat "data/full/rel2text/${SPLIT}.json" \
                | jq '.data[].in' \
                | sed 's/.$//' \
                | sed 's/^.//' \
                | sed -E 's/<(head|rel|tail)> //g' \
                    > "experiments/copy/${SPLIT}_copy.out"
        fi
        if [ -n $EVAL ]; then
            ./experiment_pipeline.sh $EVAL \
                --eval_ref "data/full/rel2text/${SPLIT}.json" \
                --experiment "copy" \
                --out_filename "${SPLIT}_copy.out" \
                --split "$SPLIT" 
        fi
        ;;
    "full_rel2text")
        ./experiment_pipeline.sh $TRAIN $DECODE $EVAL \
            --bleu_ref "data/full/rel2text/dev.json" \
            --decode_in_dir "data/full/rel2text" \
            --eval_ref "data/full/rel2text/${SPLIT}.json" \
            --experiment "full_rel2text/seed_${SEED}" \
            --out_filename "${SPLIT}_rel2text_beam${BEAM_SIZE}.out" \
            --train_in_dir "data/full/rel2text/" \
            --split "$SPLIT" \
            --beam_size "$BEAM_SIZE" \
            --seed "$SEED" \
        ;;
    "full_webnlg")
        ./experiment_pipeline.sh $TRAIN $DECODE $EVAL \
            --bleu_ref "data/ref/webnlg/dev.ref" \
            --decode_in_dir "data/full/rel2text" \
            --eval_ref "data/full/rel2text/${SPLIT}.json" \
            --experiment "full_webnlg/seed_${SEED}" \
            --out_filename "${SPLIT}_rel2text_beam${BEAM_SIZE}.out" \
            --train_in_dir "data/full/webnlg/" \
            --split "$SPLIT" \
            --beam_size "$BEAM_SIZE" \
            --seed "$SEED" \
        ;;
    "full_kelm")
        ./experiment_pipeline.sh $TRAIN $DECODE $EVAL \
            --bleu_ref "data/full/kelm/dev.json" \
            --decode_in_dir "data/full/rel2text" \
            --eval_ref "data/full/rel2text/${SPLIT}.json" \
            --experiment "full_kelm/seed_${SEED}" \
            --out_filename "${SPLIT}_rel2text_beam${BEAM_SIZE}.out" \
            --train_in_dir "data/full/kelm/" \
            --split "$SPLIT" \
            --seed "$SEED" \
            --beam_size "$BEAM_SIZE" \
            --val_check_interval 0.1 \
            --epochs 1 
        ;;
    "fewshot")
        ./experiment_pipeline.sh $TRAIN $DECODE $EVAL \
            --decode_in_dir "data/full/rel2text" \
            --eval_ref "data/full/rel2text/${SPLIT}.json" \
            --experiment "rel2text_fewshot${FEWSHOT}/seed_${SEED}" \
            --out_filename "${SPLIT}_rel2text_fewshot${FEWSHOT}_seed${SEED}_beam${BEAM_SIZE}.out" \
            --train_in_dir "data/fewshot/rel2text/${SEED}/${FEWSHOT}" \
            --split "$SPLIT" \
            --beam_size "$BEAM_SIZE" \
            --seed "$SEED" \
        ;;
    "mask_train")
        ./experiment_pipeline.sh $TRAIN $DECODE $EVAL \
            --bleu_ref "data/mask_rel/rel2text/dev.json" \
            --decode_in_dir "data/full/rel2text" \
            --eval_ref "data/full/rel2text/${SPLIT}.json" \
            --experiment "mask_train/seed_${SEED}" \
            --out_filename "${SPLIT}_rel2text_mask_train_beam${BEAM_SIZE}.out" \
            --train_in_dir "data/mask_rel/rel2text/" \
            --split "$SPLIT" \
            --beam_size "$BEAM_SIZE" \
            --seed "$SEED" \
        ;;
    "mask_test")
        ./experiment_pipeline.sh $TRAIN $DECODE $EVAL \
            --bleu_ref "data/full/rel2text/dev.json" \
            --decode_in_dir "data/mask_rel/rel2text" \
            --eval_ref "data/mask_rel/rel2text/${SPLIT}.json" \
            --experiment "mask_test/seed_${SEED}" \
            --out_filename "${SPLIT}_rel2text_mask_test_beam${BEAM_SIZE}.out" \
            --train_in_dir "data/full/rel2text/" \
            --split "$SPLIT" \
            --beam_size "$BEAM_SIZE" \
            --seed "$SEED" \
        ;;
    "mask_all")
        ./experiment_pipeline.sh $TRAIN $DECODE $EVAL \
            --bleu_ref "data/mask_rel/rel2text/dev.json" \
            --decode_in_dir "data/mask_rel/rel2text" \
            --eval_ref "data/mask_rel/rel2text/${SPLIT}.json" \
            --experiment "mask_all/seed_${SEED}" \
            --out_filename "${SPLIT}_rel2text_mask_all_beam${BEAM_SIZE}.out" \
            --train_in_dir "data/mask_rel/rel2text/" \
            --split "$SPLIT" \
            --beam_size "$BEAM_SIZE" \
            --seed "$SEED" \
        ;;
    "desc_repl")
        ./experiment_pipeline.sh $TRAIN $DECODE $EVAL \
            --bleu_ref "data/desc_repl/rel2text/dev.json" \
            --decode_in_dir "data/desc_repl/rel2text" \
            --eval_ref "data/desc_repl/rel2text/${SPLIT}.json" \
            --experiment "desc_repl/seed_${SEED}" \
            --out_filename "${SPLIT}_desc_repl_beam${BEAM_SIZE}.out" \
            --train_in_dir "data/desc_repl/rel2text" \
            --split "$SPLIT" \
            --beam_size "$BEAM_SIZE" \
            --seed "$SEED" \
        ;;
    "desc_cat")
        ./experiment_pipeline.sh $TRAIN $DECODE $EVAL \
            --bleu_ref "data/desc_cat/rel2text/dev.json" \
            --decode_in_dir "data/desc_cat/rel2text" \
            --eval_ref "data/desc_cat/rel2text/${SPLIT}.json" \
            --experiment "desc_cat/seed_${SEED}" \
            --out_filename "${SPLIT}_desc_cat_beam${BEAM_SIZE}.out" \
            --train_in_dir "data/desc_cat/rel2text" \
            --split "$SPLIT" \
            --beam_size "$BEAM_SIZE" \
            --seed "$SEED" \
        ;;
    *)
        echo "Unknown option: $1"
        ;;
esac


