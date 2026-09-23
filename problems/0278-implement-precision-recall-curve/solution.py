import numpy as np

def precision_recall_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute precision-recall pairs for different probability thresholds.
    
    Args:
        y_true: List of true binary labels (0 or 1)
        y_scores: List of predicted probabilities or confidence scores
    
    Returns:
        Tuple of (precisions, recalls, thresholds) where each is a list
    """
    
    recs=[]
    precs=[]
    y_scores=np.array(y_scores)
    y_true=np.array(y_true)
    ths= np.sort(np.unique(y_scores))[::-1].tolist()
    for th in ths:
        tp=np.sum((y_scores>=th) & y_true)
        fn= np.sum((y_scores<th) & y_true==1)
        pr= float(tp/np.sum(y_scores>=th)) if np.sum(y_scores>=th) else 0.
        rec= float(np.sum(tp/(tp+fn))) if tp+fn else 0.
        precs.append(pr)
        recs.append(rec)
    return precs,recs,ths

    