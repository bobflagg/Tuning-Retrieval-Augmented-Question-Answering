#!/usr/bin/env bash

export ROOT=/opt/enrichment/github/Tuning-Retrieval-Augmented-Question-Answering
export DATA_PATH=${ROOT}/data/train-test-df.csv
export PREFIX_PATH=${ROOT}/data/leval-1-prompt-prefix.txt
export OUTPUT_DIRECTORY=${ROOT}/model

python train.py  --fp16 --fold $1 --data_path $DATA_PATH --prefix_path $PREFIX_PATH --output_directory $OUTPUT_DIRECTORY
