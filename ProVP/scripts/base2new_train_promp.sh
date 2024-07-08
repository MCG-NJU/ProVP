#!/bin/bash

cd ..
# custom config
DATA=/storage/data1/xuchen/DATASET
TRAINER=ProMP

DATASET=$1
CFG=$2 # config file
CTP=end  # class token position (end or middle)
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
LAMBDA=1.0

for SEED in 1 2 3
do
    DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        LOSS.LAMBDA ${LAMBDA} \
        TRAINER.COOP.N_CTX 2 \
        TRAINER.COOP.CSC False \
        TRAINER.COOP.CLASS_TOKEN_POSITION end \
        DATASET.NUM_SHOTS 16 \
        DATASET.SUBSAMPLE_CLASSES base
    fi
done
