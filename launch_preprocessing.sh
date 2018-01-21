#!/usr/bin/env bash
PY_PATH="/home/anya/tensorflow/bin/python3"

TRAIN_DATA="/home/anya/toxic-comments-rep/data/train.csv"
TEST_DATA="/home/anya/toxic-comments-rep/data/test.csv"

OUTPUT_FILE="/home/anya/toxic-comments-rep/data/results.csv"
EMBEDS_FILE="/home/anya/toxic-comments-rep/data/glove.840B.100d.txt"

SWEAR_FILE="/home/anya/toxic-comments-rep/data/swear_words.csv"
WRONG_WORDS_FILE="/home/anya/toxic-comments-rep/data/correct_words.csv"


$PY_PATH main.py --train=$TRAIN_DATA --test=$TEST_DATA --output $OUTPUT_FILE --embeds $EMBEDS_FILE --swear-words=$SWEAR_FILE --wrong-words=$WRONG_WORDS_FILE
