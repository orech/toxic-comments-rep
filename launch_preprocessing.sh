#!/usr/bin/env bash
PY_PATH="/home/username/tensorflow/bin/python3"

TRAIN_DATA="/home/username/toxic-comments-rep/data/train.csv"
TEST_DATA="/home/username/toxic-comments-rep/data/test.csv"

TRAIN_CLEAN="/home/username/toxic-comments-rep/data/train.clean.npy"
TEST_CLEAN="/home/username/toxic-comments-rep/data/test.clean.npy"

EMBEDS_FILE="/home/username/toxic-comments-rep/data/crawl-300d-2M.vec"

EMBEDS_TYPE="ft_comm_crawl"
EMBEDS_CLEAN="/home/username/toxic-comments-rep/data/embeds.clean.npy"

SWEAR_FILE="/home/username/toxic-comments-rep/data/swear_words.csv"
WRONG_WORDS_FILE="/home/username/toxic-comments-rep/data/correct_words.csv"


$PY_PATH preprocessing.py --train=$TRAIN_DATA --test=$TEST_DATA --swear-words=$SWEAR_FILE --embeds=$EMBEDS_FILE --embeds-type=$EMBEDS_TYPE --embeds-clean=$EMBEDS_CLEAN --wrong-words=$WRONG_WORDS_FILE --train-clean=$TRAIN_CLEAN --test-clean=$TEST_CLEAN
