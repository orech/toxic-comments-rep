import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score, f1_score, precision_score, recall_score


def calc_metrics(y_true, y_hat, max_steps=1000):
    y_true = np.array(y_true)
    y_hat = np.array(y_hat)
    ll = float(log_loss(y_true, y_hat))
    auc = roc_auc_score(y_true, y_hat)
    f1 = []
    precision = []
    recall = []
    for i in range(1, max_steps):
        threshold = float(i) / max_steps
        y_tmp = y_hat > threshold
        f1.append(f1_score(y_true, y_tmp))
        precision.append(precision_score(y_true, y_tmp))
        recall.append(recall_score(y_true, y_tmp))
    max_idx = np.argmax(f1)
    f1 = f1[max_idx]
    precision = precision[max_idx]
    recall = recall[max_idx]
    return ll, auc, f1, precision, recall, float(max_idx + 1) / max_steps


def get_metrics(y_true, y_pred, target_labels, hist=None, plot=False):
    metrics = {}
    for i, label in enumerate(target_labels):
        metrics[label] = calc_metrics(np.array(y_true)[:, i], y_pred[:, i])
    metrics['Avg logloss'] = np.mean([metric[0] for label, metric in metrics.items()])
    if plot and hist is not None:
        plt.figure()
        plt.plot(hist.history['loss'], lw=2.0, color='b', label='train')
        plt.plot(hist.history['val_loss'], lw=2.0, color='r', label='val')
        plt.title('Sentiment')
        plt.xlabel('Epochs')
        plt.ylabel('Cross-Entropy Loss')
        plt.legend(loc='upper right')
        plt.show()
    return metrics
