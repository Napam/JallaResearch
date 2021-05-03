import numpy as np 
from matplotlib import pyplot as plt

class PCA:
    def fit(self, X):
        
        '''
        fits sorted eigenvalues and eigenvectors to class attributes. same goes for variance and explained variance.
        '''
        
        n_samples = X.shape[0]
        # We center the data and compute the sample covariance matrix.
        X -= np.mean(X, axis=0)
        self.cov_matrix_ = np.dot(X.T, X) / (n_samples-1)
        #test = np.cov(X)
        
        #Negative values are ignored with eigh
        (self.eigvalues_, self.components_) = np.linalg.eigh(self.cov_matrix_)
        
        idx = self.eigvalues_.argsort()[::-1]   
        self.eigvalues_ = self.eigvalues_[idx]
        self.components_ = self.components_[idx]
        self.variance_ = np.sum(self.eigvalues_)
        self.explained_variance_ = self.eigvalues_ / self.variance_
        
    def transform(self, X):
        #project data onto eigenvectors
        print(self.components_.shape, X.shape)
        self.projected_ = X @ self.components_.T
        return self.projected_

pca = PCA()

# Generate some dummy data
subsample = np.random.randn(69,2)*0.1
subsample[:,0] = subsample[:,0]*8
subsample[:,1] = subsample[:,0] + subsample[:,1]

pca.fit(subsample)

plt.scatter(subsample[:,0], subsample[:,1], edgecolor='none', alpha=0.5)
plt.quiver(pca.components_[0,0]*2, pca.components_[0,1]*2, 
       angles='xy', scale_units='xy', scale=1, width=0.006)
plt.quiver(pca.components_[1,0]*2, pca.components_[1,1]*2, 
       angles='xy', scale_units='xy', scale=1, width=0.006)
plt.gca().set_aspect('equal')
plt.show()