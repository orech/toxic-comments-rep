# Kaggle Competition: Toxic Comment Classification Challenge
## Table of contents

* [Introduction](#introduction)
* [Usage](#usage)
* [Results](#results)
   * [Single models](#single-models)
   * [Ensembling methods](#ensembling-methods)


## Introduction
This repository was developed by the team Toxic Random Submission as part of Kaggle Toxic comment classification challenge https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge.

It contains different models for text classification and their ensemblings, which helped us to get into **Top 3%**  :hatched_chick: on private LB.

This framework is based on the open-sourced work of two other competitors:
 - https://github.com/Donskov7/toxic_comments
 - https://github.com/PavelOstyakov/toxic
## Usage

In order to test any of the models, use your own data or take training/test datasets from the competition.
You can download training and test data from kaggle https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data and put it into the `data/` folder.
 
Firstly do the preprocessing job by running the script `launch_preprocessing.sh`. Don't forget to change paths there to your local paths. This script will tokenize your data, extract embedding vectors and save them as numpy arrays. It will also extract out of vocabulary words(those which occur in training/test data, but there are no embedding vectors for them) and save them in `data/` folder. If your data/embeddings don't change, then this should be run only once. Here is the structure of the script:


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


$PY_PATH preprocessing.py --train=$TRAIN_DATA --test=$TEST_DATA --swear-words=$SWEAR_FILE --embeds=$EMBEDS_FILE --embeds-type=$EMBEDS_TYPE --embeds-clean=$EMBEDS_CLEAN --wrong-words=$WRONG_WORDS_FILE --train-clean=$TRAIN_CLEAN --test-clean=$TEST_CLEAN

````


In order to train and directly test any of the models adapt paths and then run `launch_model_training.sh` script. `TRAIN_CLEAN`, `TEST_CLEAN`, `EMBEDS_CLEAN`, `TRAIN_LABELS` contain the preprocessed data in `.npy` format. There files should have been created by `launch_preprocessing.sh` script. The structure of `launch_model_training.sh`:


```` bash
#!/usr/bin/env bash
# Path to your tf python location
PY_PATH="/home/username/tensorflow/bin/python3"


TEST_DATA="/home/username/toxic-comments-rep/data/your_test_data.csv"

TRAIN_CLEAN="/home/username/toxic-comments-rep/path_to_preprocessed_train_data/train.clean.npy"
TEST_CLEAN="/home/username/toxic-comments-rep/path_to_preprocessed_test_data/test.clean.npy"
EMBEDS_CLEAN="/home/username/toxic-comments-rep/path_to_preprocessed_embeddings/embeds.clean.npy"
TRAIN_LABELS="/home/username/toxic-comments-rep/path_to_text_labels/train.labels.npy"

EMBEDS_TYPE="ft_comm_crawl"

LOG_FILE="log.train.BiSRU"
CONFIG="config/config.BiSRU_attention.json"


$PY_PATH train_model.py --test=$TEST_DATA --embeds_type=$EMBEDS_TYPE --train-clean=$TRAIN_CLEAN --test-clean=$TEST_CLEAN --embeds-clean=$EMBEDS_CLEAN --train-labels=$TRAIN_LABELS --config=$CONFIG --logger=$LOG_FILE

````

The variable `CONFIG` should be assigned with the path to the model config file, which is in `.json` format. In order to change model parameters, modify this config. Configs to existing models are stored in `config/` folder.


## Results

### Single models

Results presented in the table below were achieved with fastext word vectors pre-trained on Common Crawl dataset.

| Architecture            | CV score | Public score | Private score | 
| :---------------------- | -------- | ------------ |  -----------: |
| 2bi-lstm with attention | 0.9898   | 0.9867       | 0.9864        |
| capsule-net with gru    | 0.9891   | 0.9865       | 0.9860        |
| 2bi-gru with attention  | 0.9896   | 0.9864       | 0.9859        |
| 2bi-sru with attention  | 0.9901   | 0.9867       | 0.9857        |
| 2bi-gru with max-pool   | 0.9892   | 0.9861       | 0.9854        |
| dpcnn                   | 0.9890   | 0.9855       | 0.9844        | 
| simple cnn              | 0.9889   | 0.9849       | 0.9842        |

### Ensembling methods
Different meta-modelling approaches were investigated to combine single models trained on two types of word embeddings: fasttext Common Crawl and fasttext Wikipedia with generated OOV vectors.

| Model                    | CV score | Public score | Private score | 
| :----------------------- | -------- | ------------ |  -----------: |
| weighted arithmetic mean | 0.9920   | 0.9876       | 0.9868        |
| arithmetic mean          | 0.9919   | 0.9875       | 0.9867        |
| xgboost                  | 0.9915   | 0.9875       | 0.9867        |
| LightGBM                 | 0.9915   | 0.9874       | 0.9867        |
| catboost                 | 0.9917   | 0.9873       | 0.9867        |
| logistic regression      | 0.9915   | 0.9875       | 0.9863        | 
