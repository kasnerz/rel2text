#!/bin/bash


echo "=========================="
echo "Preprocessing datasets"
echo "=========================="

# Rel2Text
# for MODE in "full" "desc_cat" "desc_repl" "mask_rel"; do
#     ./preprocess.py \
#         --dataset rel2text \
#         --dataset_dir data/orig/rel2text \
#         --output data/${MODE}/rel2text \
#         --split train dev test  \
#         --mode $MODE
# done

# Rel2Text - fewshot
# for SEED in 1 2 3 4 5; do 
#     ./preprocess.py \
#         --dataset rel2text  \
#         --dataset_dir data/orig/rel2text  \
#         --output data/fewshot/rel2text/  \
#         --split train dev test   \
#         --mode full  \
#         --fewshot_splits 25 50 100 200 \
#         --seed $SEED
# done


# WebNLG
# ./preprocess.py \
#     --dataset webnlg_v3 \
#     --output data/full/webnlg \
#     --split train dev test  \
#     --mode full

# ./preprocess.py \
#     --dataset "webnlg_v3" \
#     --output data/ref/webnlg \
#     --mode full \
#     --splits test dev \
#     --extract_refs

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
    "desc_cat"
    # "full_webnlg" \
    # "full_rel2text" \
    # "full_kelm" \
    # "fewshot" \
    # "mask_train" \
    # "mask_test" \
    # "mask_all" \
    # "desc_repl" \
do
    echo "====> $EXPERIMENT"

    for SEED in 1 ; do  
        # qsub -q 'gpu*' -pe smp 8 -l 'mem_free=8G,gpu=1,gpu_ram=24G,hostname=*dll*' -cwd -pty yes -j y -b y -o out -e out -N "tde-${EXPERIMENT}-s${SEED}"\
         ./run_experiment.sh -t -d -e -s $SEED $EXPERIMENT
    done
done

# for EXPERIMENT in \
#      # rel2text_fewshot25\
#      # rel2text_fewshot50\
#      # rel2text_fewshot100\
#      # rel2text_fewshot200\
# do
#     echo "====> $EXPERIMENT"
#     for SEED in 1 2 3 4 5; do  
#         # for FEWSHOT in 25 50 100 200; do 
#         #      qsub -q 'gpu*' -pe smp 8 -l 'mem_free=8G,gpu=1,gpu_ram=24G,hostname=*dll*' -cwd -pty yes -j y -b y -o out -e out -N "tde-${EXPERIMENT}_${FEWSHOT}-s${SEED}"\
#         #       ./run_experiment.sh -t -d -e -s $SEED -f $FEWSHOT $EXPERIMENT
#         # done

#         EVAL_FILE=$(ls experiments/$EXPERIMENT/seed_${SEED}/*.eval.json)
#         for METRIC in "bleu" "meteor" "bleurt" "nubia_semsim" "nubia_contr" "nubia_neutr" "nubia_agree" "nubia" "unique-1" "cond_entropy-2" "msttr-100" "nubia_perpl"; do
#             RES=$(jq ".\"${METRIC}\"" "$EVAL_FILE")

#             if [ $METRIC = "meteor" ]; then 
#                 RES=$(echo "$RES * 100" | bc)
#             fi
#             printf "%.2f;" "$RES"
#         done

#         RES=$(jq ".\"mean_pred_length\"" "$EVAL_FILE")
#         printf "%.2f" "$RES"
#         echo
#     done

# done