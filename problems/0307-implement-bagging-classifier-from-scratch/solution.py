import numpy as np

def train_learner(X,y):
    best_misclas= float('inf')
    best_th= float('inf')
    best_feat= 0
    for f in range(X.shape[1]):
        ths= np.unique(X[:,f])
        for th in ths:
            preds= X[:,f]>th
            error= len(y)- np.sum(preds==y)
            if error<best_misclas:
                best_misclas= error
                best_feat= f
                best_th= th
    return {
        'feature': best_feat,
        'threshold': best_th
    }

                


def bagging_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_estimators: int = 10, seed: int = 42) -> np.ndarray:
    """
    Implement a bagging classifier using decision stumps.
    
    Args:
        X_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,), binary {0, 1}
        X_test: Test features of shape (n_test_samples, n_features)
        n_estimators: Number of bootstrap samples/base estimators
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray: Predicted labels for X_test
    """
    # Your code here
    rng = np.random.default_rng(seed=seed)
    n, _= X_train.shape
    indices= np.arange(n)
    bootsraps_ind=[
        rng.choice(indices, size= n) for _ in range(n_estimators)
    ]
    Xt_boots= [
        X_train[ind] for ind in bootsraps_ind
    ]
    yt_boots= [
        y_train[ind] for ind in bootsraps_ind
    ]
    learners= []
    for i in range(n_estimators):
        learner= train_learner(Xt_boots[i], yt_boots[i])
        learners.append(learner)

    boots_preds= np.array([X_test[:,learner['feature']]>learner['threshold'] for learner in learners])
    # Rows are bootstraps preds 
    return np.int64((np.sum(boots_preds,axis=0)/n_estimators)>=0.5)
    

