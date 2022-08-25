"""Module level docstring
"""
from sklearn import metrics
import matplotlib.pyplot as plt

def get_auroc(y_test, y_pred_proba, plot=False):
    """Plots the roc curve and returns auroc

    THIS CODE IS FROM: VICTOR (KAIYANG) CHEN.
    He's a great guy and an excellent developer
    Thank you, Victor.
    """
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    if plot:

        #create ROC curve
        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(fpr,tpr,label="AUC="+str(auc))
        ax.set_ylabel('True Positive Rate', fontsize=24)
        ax.set_xlabel('False Positive Rate', fontsize=24)
        ax.legend(loc=4, fontsize=16)
    return auc, fig
