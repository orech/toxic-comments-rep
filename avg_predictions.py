import pandas as pd
import argparse
import os.path
from os import listdir
import numpy as np

target_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
num_classes = len(target_labels)
num_test_samples = 153164

def main():
    df = pd.read_csv('../submissions/original_pyramidCNN_folds.csv')
    predictions_path = './attention_spat_dropout'
    result_path = './outputs/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    final_predictions = np.ones([num_test_samples, num_classes])
    i = 0
    for file in listdir(predictions_path):
        file = os.path.join(predictions_path,file)
        probs = np.load(file)
        final_predictions *= probs
        i += 1
    final_predictions = final_predictions ** (1/i)
    print(i)

    test_ids = df["id"].values
    test_ids = test_ids.reshape((len(test_ids), 1))

    test_predicts = pd.DataFrame(data=final_predictions, columns=target_labels)
    test_predicts["id"] = test_ids
    test_predicts = test_predicts[["id"] + target_labels]
    submit_path = os.path.join(result_path, "{0}.csv".format("attention_spat_dropout_folds"))
    test_predicts.to_csv(submit_path, index=False)
if __name__=='__main__':
    main()