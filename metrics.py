import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score


def calc_metrics(y_true, y_hat, max_steps=1000):
    y_true = np.array(y_true)
    y_hat = np.array(y_hat)
    metrics = {}
    metrics['Logloss'] = float(log_loss(y_true, y_hat))
    metrics['AUC'] = roc_auc_score(y_true, y_hat)
    metrics['F1'] = []
    metrics['Precision'] = []
    metrics['Recall'] = []
    for i in range(1, max_steps):
        threshold = float(i) / max_steps
        y_tmp = y_hat > threshold
        metrics['F1'].append(f1_score(y_true, y_tmp))
        metrics['Precision'].append(precision_score(y_true, y_tmp))
        metrics['Recall'].append(recall_score(y_true, y_tmp))
    max_idx = np.argmax(metrics['F1'])
    metrics['F1'] = metrics['F1'][max_idx]
    metrics['Precision'] = metrics['Precision'][max_idx]
    metrics['Recall'] = metrics['Recall'][max_idx]
    metrics['Threshold'] = float(max_idx + 1) / max_steps
    return metrics


def get_metrics(y_true, y_pred, target_labels, hist=None, plot=False):
    metrics = {}
    for i, label in enumerate(target_labels):
        metrics[label] = calc_metrics(np.array(y_true)[:, i], y_pred[:, i])
    metrics['Avg'] = {'Logloss': np.mean([metric['Logloss'] for label, metric in metrics.items()])}
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


def print_metrics(metrics):
    result_str = []
    for label, metric in metrics.items():
        metric_str = []
        for metric_name, value in metric.items():
            metric_str.append('{} = {}'.format(metric_name, value))
        metric_str = '\n\t'.join(metric_str)
        result_str.append('{}\n\t{}'.format(label, metric_str))
    return '\n'.join(result_str)
