import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def _save_fig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

def evaluate_and_plot(df: pd.DataFrame, results: dict, figs_dir: str):
    # Sentiment distribution (VADER if present)
    if 'vader_compound' in df.columns:
        plt.figure()
        df['vader_compound'].hist(bins=20)
        plt.title('VADER Compound Score Distribution')
        plt.xlabel('compound')
        plt.ylabel('count')
        _save_fig(os.path.join(figs_dir, 'vader_compound_hist.png'))

    # If supervised, plot confusion matrix
    if results.get('mode') == 'supervised_lr' and 'y_test' in results:
        y_test = np.array(results['y_test'])
        y_pred = np.array(results['y_pred'])
        cm = confusion_matrix(y_test, y_pred, labels=[0,1,2])
        plt.figure()
        plt.imshow(cm, interpolation='nearest')
        plt.title('Confusion Matrix (0=neg,1=neu,2=pos)')
        plt.colorbar()
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha='center', va='center')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        _save_fig(os.path.join(figs_dir, 'confusion_matrix.png'))

        # metrics text file
        report = classification_report(y_test, y_pred, target_names=['neg','neu','pos'])
        with open(os.path.join(figs_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
