#!/usr/bin/env bash
# Path to tf python location
PY_PATH="/home/username/tensorflow/bin/python3"

TRAIN_DATA="/home/username/toxic-comments-rep/data/train.csv"
TEST_DATA="/home/username/toxic-comments-rep/data/test.csv"

TRAIN_CLEAN="/home/username/toxic-comments-rep/data/train.clean.npy"
TEST_CLEAN="/home/username/toxic-comments-rep/data/test.clean.npy"

EMBEDS_FILE="/home/username/toxic-comments-rep/data/crawl-300d-2M.vec"

EMBEDS_TYPE="ft_comm_crawl"
EMBEDS_CLEAN="/home/username/toxic-comments-rep/data/embeds.clean.npy"


$PY_PATH preprocessing.py --train=$TRAIN_DATA --test=$TEST_DATA --embeds=$EMBEDS_FILE --embeds-type=$EMBEDS_TYPE --embeds-clean=$EMBEDS_CLEAN --train-clean=$TRAIN_CLEAN --test-clean=$TEST_CLEAN
