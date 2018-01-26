#!/usr/bin/env bash
#!/usr/bin/env bash
PY_PATH="/home/anya/tensorflow/bin/python3"

TRAIN_DATA="/home/anya/toxic-comments-rep/data/train.clean.csv"
TEST_DATA="/home/anya/toxic-comments-rep/data/test.clean.csv"

OUTPUT_FILE="/home/anya/toxic-comments-rep/data/results.csv"
EMBEDS_FILE="/home/anya/toxic-comments-rep/data/wiki.en.vec"
# fasttext, glove or word2vec
EMBEDS_TYPE="fasttext"

SWEAR_FILE="/home/anya/toxic-comments-rep/data/swear_words.csv"
WRONG_WORDS_FILE="/home/anya/toxic-comments-rep/data/correct_words.csv"

CONFIG="config.json"


$PY_PATH train_model.py --train=$TRAIN_DATA --test=$TEST_DATA --embeds=$EMBEDS_FILE --embeds_type=$EMBEDS_TYPE --swear-words=$SWEAR_FILE --wrong-words=$WRONG_WORDS_FILE --config=$CONFIG --output=$OUTPUT_FILE
