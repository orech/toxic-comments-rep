import pandas as pd
import argparse
import os.path
from os import listdir
import numpy as np

target_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
num_classes = len(target_labels)
num_test_samples = 153164

def main():

    submission_path = '../blending'
    attention = pd.read_csv('../blending/attention_spat_dropout_folds.csv')
    gru = pd.read_csv('../blending/2BiGRU_rec_dropout_glob_max_pool_fasttext_folds.csv')
    pyramid = pd.read_csv('../blending/pyramidCNN_fasttext_folds.csv')
    cnn = pd.read_csv('../blending/simple_cnn_spat_drop_folds.csv')

    result_path = './blending_results/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)


    final_predictions = ( attention[target_labels].values ** 3 * gru[target_labels].values ** 2 * pyramid[target_labels].values ** 2 * cnn[target_labels].values) ** (1/8)

    test_ids = attention["id"].values
    test_ids = test_ids.reshape((len(test_ids), 1))

    test_predicts = pd.DataFrame(data=final_predictions, columns=target_labels)
    test_predicts["id"] = test_ids
    test_predicts = test_predicts[["id"] + target_labels]
    submit_path = os.path.join(result_path, "{0}.csv".format("weighted_geom_avg"))
    test_predicts.to_csv(submit_path, index=False)
if __name__=='__main__':
    main()
