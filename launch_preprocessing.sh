#!/usr/bin/env bash
PY_PATH="/home/anya/tensorflow/bin/python3"

TRAIN_DATA="/home/anya/toxic-comments-rep/data/train.csv"
TEST_DATA="/home/anya/toxic-comments-rep/data/test.csv"

TRAIN_CLEAN="/home/anya/toxic-comments-rep/data/train.clean.csv"
TEST_CLEAN="/home/anya/toxic-comments-rep/data/test.clean.csv"

SWEAR_FILE="/home/anya/toxic-comments-rep/data/swear_words.csv"
WRONG_WORDS_FILE="/home/anya/toxic-comments-rep/data/correct_words.csv"


$PY_PATH preprocessing.py --train=$TRAIN_DATA --test=$TEST_DATA --swear-words=$SWEAR_FILE --wrong-words=$WRONG_WORDS_FILE --train-clean=$TRAIN_CLEAN --test-clean=$TEST_CLEAN
