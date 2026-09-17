import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        self.smoothing=smoothing
        
    def forward(self, X, y):
        # Fit model to binary features X and labels y
        _, counts= np.unique(y, return_counts=True)
        self.prior= counts/(np.size(y))
        like_1= np.array((np.sum(X[y==1],axis=0)+self.smoothing)/(X.shape[0]+2*self.smoothing)).reshape(-1,1)
        like_0= np.array((np.sum(X[y==0],axis=0)+self.smoothing)/(X.shape[0]+2*self.smoothing)).reshape(-1,1)
        self.probas= np.log(np.hstack((like_1,like_0)))
        
    def predict(self, X):
        # Predict class labels for test set X
        _= self.prior* (np.dot(X,self.probas))
        return np.int64(_[:,0]>_[:,1])