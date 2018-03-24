# Kaggle Competition: Toxic Comment Classification Challenge
This repository contains the code for Kaggle competition https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge


This framework contains different models for text classification, which helped us to get into Top 3% on private LB.

In order to test any of the models, use your own data or take training/test datasets from the competition.
You can download training and test data from kaggle https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data and put it into the `data/` folder.
 
Firstly do the preprocessing job by running the script `launch_preprocessing.sh`. Don't forget to change paths there to your local paths. This script will tokenize your data, extract embedding vectors and save them as numpy arrays. Here is the structure of the script:

```` bash
#!/usr/bin/env bash
# Path to your tf python location
PY_PATH="/home/username/tensorflow/bin/python3"

TRAIN_DATA="/home/username/toxic-comments-rep/data/your_training_data.csv"
TEST_DATA="/home/username/toxic-comments-rep/data/your_test_data.csv"

TRAIN_CLEAN="/home/username/toxic-comments-rep/where_to_save_preprocessed_train_data/train.clean.npy"
TEST_CLEAN="/home/username/toxic-comments-rep/where_to_save_preprocessed_test_data/test.clean.npy"

EMBEDS_FILE="/home/username/toxic-comments-rep/path_to_word_embeddings/embeds.vec"

EMBEDS_TYPE="ft_comm_crawl"
EMBEDS_CLEAN="/home/username/toxic-comments-rep/where_to_save_embedding_vectors/embeds.clean.npy"

# you can also use these files to replace misspelled swear words in your data
SWEAR_FILE="/home/username/toxic-comments-rep/path_to_swear_words/swear_words.csv"
WRONG_WORDS_FILE="/home/username/toxic-comments-rep/path_to_correct_orthography/correct_words.csv"


$PY_PATH preprocessing.py --train=$TRAIN_DATA --test=$TEST_DATA --swear-words=$SWEAR_FILE --embeds=$EMBEDS_FILE --embeds-type=$EMBEDS_TYPE --embeds-clean=$EMBEDS_CLEAN --wrong-words=$WRONG_WORDS_FILE --train-clean=$TRAIN_CLEAN --test-clean=$TEST_CLEAN

````
