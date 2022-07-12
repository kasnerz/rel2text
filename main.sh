#!/bin/bash


echo "=========================="
echo "Preprocessing datasets"
echo "=========================="

# Rel2Text
# currently filters relations for every setup: TODO speedup
for MODE in "full" "desc_cat" "desc_repl" "mask_rel"; do
    ./preprocess.py \
        --dataset rel2text \
        --dataset_dir data/orig/rel2text \
        --output data/${MODE}/rel2text \
        --split train dev test  \
        --mode $MODE
done

# Rel2Text - fewshot
for SEED in 1 2 3 4 5; do
    ./preprocess.py \
        --dataset rel2text  \
        --dataset_dir data/orig/rel2text  \
        --output data/fewshot/rel2text/  \
        --split train dev test   \
        --mode full  \
        --fewshot_splits 25 50 100 200 \
        --seed $SEED
done

# WebNLG
./preprocess.py \
    --dataset webnlg_v3 \
    --output data/full/webnlg \
    --split train dev test  \
    --mode full

./preprocess.py \
    --dataset "webnlg_v3" \
    --output data/ref/webnlg \
    --mode full \
    --splits test dev \
    --extract_refs

# # KeLM
# # Uncomment these lines for preprocessing the KeLM dataset.
# # Note that preprocessing may take a long time.
# ./preprocess.py \
#     --dataset kelm \
#     --dataset_dir data/orig/kelm \
#     --output data/full/kelm \
#     --split train dev test  \
#     --mode full


echo "=========================="
echo "Running experiments"
echo "=========================="

for EXPERIMENT in  \
    "full_rel2text" \
    "full_webnlg" \
    # "full_kelm" \
    "mask_train" \
    "mask_test" \
    "mask_all" \
    "desc_cat" \
    "desc_repl"
do
    echo "====> $EXPERIMENT"
    for SEED in 1 2 3 4 5; do
        # qsub -q 'gpu*' -pe smp 8 -l 'mem_free=8G,gpu=1,gpu_ram=24G,hostname=*dll*' -cwd -pty yes -j y -b y -o out -e out -N "tde-${EXPERIMENT}-s${SEED}"\
         ./run_experiment.sh -t -d -e -s $SEED $EXPERIMENT
    done
done

for SEED in 1 2 3 4 5; do
    for FEWSHOT in 25 50 100 200; do
        echo "====> fewshot${FEWSHOT}"
         # qsub -q 'gpu*' -pe smp 8 -l 'mem_free=8G,gpu=1,gpu_ram=24G,hostname=*dll*' -cwd -pty yes -j y -b y -o out -e out -N "tde-${EXPERIMENT}_${FEWSHOT}-s${SEED}"\
          ./run_experiment.sh -t -d -e -s $SEED -f $FEWSHOT "fewshot"
    done
done
