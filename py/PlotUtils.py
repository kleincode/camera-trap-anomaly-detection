# This file defines helper functions for plotting.
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(test_labels: list, test_df: list, title: str, figsize=(8, 8), savefile = None, show: bool = True):
    """Plots the roc curve of a classifier.

    Args:
        test_labels (list): Labels for the test examples.
        test_df (list): Decision function values for the test examples.
        title (str): Title of the plot.
        figsize (tuple, optional): Size of the plot. Defaults to (8, 8).
        savefile (_type_, optional): Output file without ending. Will be saved as pdf and png. If None, the plot is not saved. Defaults to None.
        show (bool, optional): If False, do not show the plot. Defaults to True.

    Returns:
        fpr (list of float), tpr (list of float), thresholds (list of float), auc_score (float): Points on roc curves, their thresholds, and the area under ROC curve.
    """
    fpr, tpr, thresholds = roc_curve(test_labels, test_df)
    auc_score = auc(fpr, tpr)

    if not show:
        plt.ioff()
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, lw=1)
    plt.fill_between(fpr, tpr, label=f"AUC = {auc_score:.4f}", alpha=0.5)
    plt.plot([0, 1], [0, 1], color="gray", linestyle="dotted")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{title}")
    plt.legend(loc="lower right")
    if savefile is not None:
        plt.savefig(f"{savefile}.png", bbox_inches="tight")
        plt.savefig(f"{savefile}.pdf", bbox_inches="tight")
    if show:
        plt.show()
    return fpr, tpr, thresholds, auc_score

def get_percentiles(fpr, tpr, thresholds, percentiles=[0.9, 0.95, 0.98, 0.99], verbose = True):
    """Returns the maximum possible TNR (elimination rate) for given minimum TPR.

    Args:
        fpr (list of float): FPR values from ROC curve.
        tpr (list of float): TPR values from ROC curve.
        thresholds (list of float): Thresholds from ROC curve.
        percentiles (list of float, optional): List of minimum TPR values to use as input. Defaults to [0.9, 0.95, 0.98, 0.99].
        verbose (bool, optional): If True, print the results. Defaults to True.

    Returns:
        list of float: TNR values aka elimination rates.
    """
    assert percentiles == sorted(percentiles)
    tnrs = []
    for percentile in percentiles:
        for i, tp in enumerate(tpr):
            if tp >= percentile:
                tnrs.append(1 - fpr[i]) # append tnr
                if verbose:
                    print(f"{percentile} percentile : TPR = {tp:.4f}, FPR = {fpr[i]:.4f} <-> TNR = {(1 - fpr[i]):.4f} @ thresh {thresholds[i]}")
                break
    return tnrs