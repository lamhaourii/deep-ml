import numpy as np

def compute_roc_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute ROC curve points (FPR, TPR) for binary classification.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted scores/probabilities for the positive class
    
    Returns:
        Tuple of (fpr, tpr) where each is a list of floats
    """
    # Your code here
    tprs=[0.]
    fprs=[0.]
    y_scores=np.array(y_scores)
    y_true=np.array(y_true)

    for th in np.unique(y_scores)[::-1]:
        tp=np.sum((y_scores>=th) & (y_true==1))
        p=np.sum(y_true)
        fp= np.sum((y_scores>=th) & (y_true==0))
        n= np.sum(y_true==0)
        tpr= float(tp/p) if p else 0.
        fpr= float(fp/n) if n else 0.
        tprs.append(tpr)
        fprs.append(fpr)
    return fprs,tprs
    