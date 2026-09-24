import numpy as np
import math
from collections import Counter



class DecisionTree:
    """
    A decision tree classifier that the harness will use as the base learner
    in a Random Forest. The harness will create many DecisionTree instances,
    train each on a different bootstrap sample of the data, and aggregate
    their predictions by majority vote.

    For the ensemble to beat a single tree by a meaningful margin, your trees
    must be DIVERSE. Bootstrap sampling (handled by the harness) gives some
    diversity. The most effective additional source of diversity is to
    randomize WHICH features each split considers -- this is what makes a
    Random Forest different from plain Bagging.

    Use `self.random_state` for any randomness inside your tree so that
    different harness seeds produce different trees.
    """

    def __init__(self, max_depth=10, min_samples_split=2,
                 max_features='sqrt', random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features      # 'sqrt', None, or int
        self.random_state = random_state
        self.rng= np.random.default_rng(seed=self.random_state)

    def _calculate_entropy(self, y) -> float:
        """Calculate the entropy of a list of labels."""
        _, counts= np.unique(y,return_counts=True )
        freqs= counts/np.size(y)
        return -np.sum(freqs*np.log(freqs))
    
    def _calculate_information_gain(self, X, y, th, feat:int) -> float:
        """Calculate the information gain of splitting on attr."""
        """ feat is the index of the feature we split on"""
        
        entropy= self._calculate_entropy(y)  
        labels= [y[X[:,feat]>=th], y[X[:,feat]<th]]
        if len(labels[0]) == 0 or len(labels[1]) == 0:
            return None
        entropies=np.array([self._calculate_entropy(lab) for lab in labels])
        freqs= np.array([len(lab) for lab in labels])/len(X)  
        return entropy - np.sum(freqs*entropies)
    
    def _majority_class(self,y) -> str:
        """Return the majority class. Break ties alphabetically."""
        ctr= Counter(y)
        ctr= sorted(ctr.items(), key=lambda pair: (-pair[1], pair[0]))
        maj= ctr[0][0]
        return maj
    
    def _learn(self, X, y, depth):
        if len(y) < self.min_samples_split:
            return self._majority_class(y)
        if depth==self.max_depth:
            return self._majority_class(y)    
        if len(set(y.tolist()))==1:
            return y[0]
        feats= np.arange(X.shape[1])
        max_gain=float('-inf')
        
        if self.max_features=='sqrt':
            n_features_to_consider=int(np.sqrt(X.shape[1]))
        elif self.max_features is None:
            n_features_to_consider= self.rng.integers(1, X.shape[1])
        else:
            n_features_to_consider= self.max_features

        
        mask = np.zeros(X.shape[1], dtype=bool)
        random_indices = self.rng.choice(X.shape[1], size= n_features_to_consider, replace=False)
        mask[random_indices] = True
        feats= feats[mask]

        best_feat=feats[0]

        for feat in feats:
            ths= np.unique(X[:, feat])
            for th in ths:
                gain= self._calculate_information_gain(X, y, th, feat)
                if gain is None:
                    continue
                if gain>max_gain:
                    max_gain, best_feat, best_th=gain, feat, th

        if max_gain == -np.inf:
            return self._majority_class(y)
        left= X[X[:,best_feat]<best_th]
        right= X[X[:,best_feat]>=best_th]
        
        return {
            'feature':best_feat,
            'threshold':best_th,
            'left':self._learn(left, y[X[:,best_feat]<best_th], depth+1),
            'right':self._learn(right, y[X[:,best_feat]>=best_th], depth+1)
        }
        
    def fit(self, X, y):
        """
        Fit the tree on (X, y).

        Args:
            X: numpy array of shape (n_samples, n_features), dtype float
            y: numpy array of shape (n_samples,), integer class labels in [0, n_classes)

        Returns:
            self
        """
        # TODO: build the tree, storing it on self
        

        self.tree= self._learn(X, y, depth=0)
    
        return self

    def predict(self, X):
        """
        Predict integer class labels for X.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with integer class labels
        """
        # TODO: traverse the tree for each row in X
        n_samples= X.shape[0]
        out= np.zeros((n_samples,))
        for i,x in enumerate(X):
            node= self.tree
            while isinstance(node, dict):
                if x[node['feature']]<node['threshold']:
                    node= node['left']
                else:
                    node= node['right']
            out[i]=node
        return out

        
