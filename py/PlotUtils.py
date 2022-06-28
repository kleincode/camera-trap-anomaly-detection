import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(test_labels: list, test_df: list, title: str, figsize=(8, 8), savefile = None, show: bool = True):
    fpr, tpr, thresholds = roc_curve(test_labels, test_df)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, lw=1, label="ROC Curve")
    plt.plot([0, 1], [0, 1], color="lime", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{title} (AUC = {auc_score})")
    plt.legend(loc="lower right")
    if savefile is not None:
        plt.savefig(f"{savefile}.png", bbox_inches="tight")
        plt.savefig(f"{savefile}.pdf", bbox_inches="tight")
    if show:
        plt.show()
    return fpr, tpr, thresholds, auc_score

def get_percentiles(fpr, tpr, thresholds, percentiles=[0.9, 0.95, 0.98, 0.99]):
    for percentile in percentiles:
        for i, tp in enumerate(tpr):
            if tp >= percentile:
                print(f"{percentile} percentile : TPR = {tp:.4f}, FPR = {fpr[i]:.4f} <-> TNR = {(1 - fpr[i]):.4f} @ thresh {thresholds[i]}")
                break