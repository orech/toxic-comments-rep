import pandas as pd
import argparse
import os.path
import numpy as np

target_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
num_classes = len(target_labels)

def main():
    print("Stsrted")
    parser = argparse.ArgumentParser(description='--gru=$GRU --cnn=$CNN --attention=$ATTEND')
    parser.add_argument('gru-submission')
    parser.add_argument('cnn-submission')
    parser.add_argument('attention-submission')
    args = parser.parse_args()

    result_path = './outputs/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    gru_sub = pd.read_csv('best_subs/2BiGRU_GlobMaxPool_folds.submit')
    cnn_sub = pd.read_csv('best_subs/original_pyramidCNN_folds.csv')
    attention_sub = pd.read_csv('best_subs/BiGRU_attention_folds.submit')
    gru_probs = np.array(gru_sub[target_labels].values)
    cnn_probs = np.array(cnn_sub[target_labels].values)
    attention_probs = np.array(attention_sub[target_labels].values)

    final_predictions = (gru_probs * cnn_probs * attention_probs) ** (1/3)


    test_ids = gru_sub["id"].values
    test_ids = test_ids.reshape((len(test_ids), 1))

    test_predicts = pd.DataFrame(data=final_predictions, columns=target_labels)
    test_predicts["id"] = test_ids
    test_predicts = test_predicts[["id"] + target_labels]
    submit_path = os.path.join(result_path, "{0}.csv".format("gru_cnn_attend_ensemble.csv"))
    test_predicts.to_csv(submit_path, index=False)
if __name__=='__main__':
    main()
