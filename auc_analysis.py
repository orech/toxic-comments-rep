import numpy as np
import pandas as pd
from bokeh.io import output_file, show, output_notebook
from bokeh.layouts import column, gridplot
from bokeh.plotting import figure
from bokeh.palettes import brewer

output_file('plot')
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold


def main():

    target_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    test_predicts_path = "./stacking/catboost_x_test.npy"
    x_test =  np.load(test_predicts_path)
    val_predicts_path = "./stacking/catboost_x_train.npy"
    x_meta = np.load(val_predicts_path)

    # ROC AUC analysis based on https://www.kaggle.com/ogrellier/things-you-need-to-be-aware-of-before-stacking
    class_preds = [c_ for c_ in target_labels]
    folds = KFold(n_splits=10, shuffle=True, random_state=1)
    oof = pd.DataFrame(data=x_meta[:,:6], columns=target_labels)
    sub = pd.DataFrame(data=x_test[:,:6], columns=target_labels)


    figures = []
    for i_class, class_name in enumerate(target_labels):
        s = figure(plot_width=600, plot_height=300,
                   title="Probability logits for %s" % class_name)

        for n_fold, (_, val_idx) in enumerate(folds.split(oof)):
            probas = oof[class_preds[i_class]].values[val_idx]
            p_log = np.log((probas + 1e-5) / (1 - probas + 1e-5))
            hist, edges = np.histogram(p_log, density=True, bins=50)
            s.line(edges[:50], hist, legend="Fold %d" % n_fold, color=brewer["Set1"][4][n_fold])
            if n_fold == 3:
                break

        oof_probas = oof[class_preds[i_class]].values
        oof_logit = np.log((oof_probas + 1e-5) / (1 - oof_probas + 1e-5))
        hist, edges = np.histogram(oof_logit, density=True, bins=50)
        s.line(edges[:50], hist, legend="Full OOF", color=brewer["Paired"][6][1], line_width=3)

        sub_probas = sub[class_name].values
        sub_logit = np.log((sub_probas + 1e-5) / (1 - sub_probas + 1e-5))
        hist, edges = np.histogram(sub_logit, density=True, bins=50)
        s.line(edges[:50], hist, legend="Test", color=brewer["Paired"][6][5], line_width=3)
        figures.append(s)

    # put the results in a column and show
    show(column(figures))


    train_y = np.load('./data/train.labels.npy')[:x_meta.shape[0]]
    roc_auc_val = []
    val_size = int(len(train_y) / 10)
    for i in range(10):
        roc_auc_val.append(roc_auc_score(train_y[i * val_size : i * val_size + val_size, :6], x_meta[i * val_size : i * val_size + val_size,:6]))
    roc_auc_mean = sum(roc_auc_val)/len(roc_auc_val)
    print('Mean auc: {0}'.format(roc_auc_mean))

    roc_auc_train = roc_auc_score(train_y[:,:6], x_meta[:,:6])
    print('OOF auc: {0}'.format(roc_auc_train))


if __name__=='__main__':
    main()
