#!/usr/bin/env bash
PY_PATH="/home/anya/tensorflow/bin/python3"

TRAIN_DATA="/home/anya/toxic-comments-rep/data/train.clean.csv"
TEST_DATA="/home/anya/toxic-comments-rep/data/test.clean.csv"

TRAIN_CLEAN="/home/anya/toxic-comments-rep/data/train.clean.csv"
TEST_CLEAN="/home/anya/toxic-comments-rep/data/test.clean.csv"

OUTPUT_FILE="/home/anya/toxic-comments-rep/data/results.csv"
EMBEDS_FILE="/home/anya/toxic-comments-rep/data/glove.twitter.27B.100d.txt"
# fasttext, glove or word2vec
EMBEDS_TYPE="glove"

SWEAR_FILE="/home/anya/toxic-comments-rep/data/swear_words.csv"
WRONG_WORDS_FILE="/home/anya/toxic-comments-rep/data/correct_words.csv"


$PY_PATH main.py --train=$TRAIN_DATA --test=$TEST_DATA --output $OUTPUT_FILE --embeds=$EMBEDS_FILE --embeds_type=$EMBEDS_TYPE --swear-words=$SWEAR_FILE --wrong-words=$WRONG_WORDS_FILE --train-clean=$TRAIN_CLEAN --test-clean=$TEST_CLEAN
