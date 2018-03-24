#!/usr/bin/env bash
#!/usr/bin/env bash
PY_PATH="/home/username/tensorflow/bin/python3"


TEST_DATA="/home/username/toxic-comments-rep/data/test.csv"
TRAIN_CLEAN="/home/username/toxic-comments-rep/data/train.clean.npy"
TEST_CLEAN="/home/username/toxic-comments-rep/data/test.clean.npy"
EMBEDS_CLEAN="/home/username/toxic-comments-rep/data/embeds.clean.npy"
TRAIN_LABELS="/home/username/toxic-comments-rep/data/train.labels.npy"

EMBEDS_TYPE="ft_comm_crawl"

SWEAR_FILE="/home/username/toxic-comments-rep/data/swear_words.csv"
WRONG_WORDS_FILE="/home/username/toxic-comments-rep/data/correct_words.csv"

LOG_FILE="log.train.BiSRU"
CONFIG="config/config.BiSRU_attention.json"


$PY_PATH train_model.py --test=$TEST_DATA --embeds_type=$EMBEDS_TYPE --train-clean=$TRAIN_CLEAN --test-clean=$TEST_CLEAN --embeds-clean=$EMBEDS_CLEAN --train-labels=$TRAIN_LABELS --config=$CONFIG --logger=$LOG_FILE
