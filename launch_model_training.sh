#!/usr/bin/env bash
#!/usr/bin/env bash
PY_PATH="/home/anya/tensorflow/bin/python3"

TRAIN_DATA="/home/anya/toxic-comments-rep/data/train.csv"
TEST_DATA="/home/anya/toxic-comments-rep/data/test.csv"

OUTPUT_FILE="/home/anya/toxic-comments-rep/data/results.csv"
EMBEDS_FILE="/home/anya/toxic-comments-rep/data/crawl-300d-2M.vec"
# fasttext, glove or word2vec
EMBEDS_TYPE="ft_comm_crawl"

TRAIN_CLEAN="/home/anya/toxic-comments-rep/data/train.clean.npy"
TEST_CLEAN="/home/anya/toxic-comments-rep/data/test.clean.npy"
EMBEDS_CLEAN="/home/anya/toxic-comments-rep/data/embeds.clean.npy"
TRAIN_LABELS="/home/anya/toxic-comments-rep/data/train.labels.npy"

SWEAR_FILE="/home/anya/toxic-comments-rep/data/swear_words.csv"
WRONG_WORDS_FILE="/home/anya/toxic-comments-rep/data/correct_words.csv"

LOG_FILE="log.train.BiSRU"

CONFIG="config.2BiSRU_GlobMaxPool.json"


$PY_PATH train_model.py --train=$TRAIN_DATA --test=$TEST_DATA --embeds=$EMBEDS_FILE --embeds_type=$EMBEDS_TYPE --train-clean=$TRAIN_CLEAN --test-clean=$TEST_CLEAN --embeds-clean=$EMBEDS_CLEAN --train-labels=$TRAIN_LABELS --config=$CONFIG --output=$OUTPUT_FILE --logger=$LOG_FILE
