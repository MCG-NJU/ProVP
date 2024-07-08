#!/bin/bash

cd ..

# custom config
DATA=/data1/xuchen/DATASET
TRAINER=ProVP

DATASET=$1
CFG=$2  # config file
NUMP=$4  # number of prompts
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
LAMBDA=1.0

for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
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
        TRAINER.COOP.N_CTX ${NUMP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done
