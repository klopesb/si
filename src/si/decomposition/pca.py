import numpy as np
from si.base.transformer import Transformer

#KB - Exercise 5 

class PCA(Transformer):
    """
    Principal Component Analysis (PCA) transformer for dimensionality reduction.
    
    PCA reduces the dimensionality of data by projecting it onto the principal components
    (directions of maximum variance). It performs eigendecomposition on the covariance matrix
    to find these components.
    
    Parameters
    ----------
    n_components : int
        Number of principal components to keep
    
    Attributes
    ----------
    mean : np.ndarray
        Mean of each feature in the training data (used for centering)
    components : np.ndarray
        Principal components (eigenvectors), shape (n_features, n_components)
    explained_variance : np.ndarray
        Proportion of variance explained by each component
    """

    def __init__(self, n_components):
        """
        Initialize the PCA transformer.
        
        Parameters
        ----------
        n_components : int
            Number of principal components to retain
        """
        super().__init__()
        self.n_components = n_components
        
        # parâmetros estimados
        self.mean = None
        self.components = None   # eigenvectors
        self.explained_variance = None  # eigenvalues / total

    def _fit(self, dataset):
        """
        Fit the PCA model by computing principal components from the training data.
        
        The fitting process:
        1. Centers the data by subtracting the mean
        2. Computes the covariance matrix
        3. Performs eigendecomposition
        4. Sorts eigenvectors by eigenvalues (descending)
        5. Selects the top n_components
        6. Calculates explained variance ratios
        
        Parameters
        ----------
        X : np.ndarray
            Training data matrix, shape (n_samples, n_features)
        
        Returns
        -------
        self : PCA
            The fitted transformer instance
        """
        X = dataset.X 
        #Center the data 
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        #Covariance Matrix
        covariance_matrix = np.cov(X_centered, rowvar=False, ddof=1)

        #Eigen Decomposition
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        #Sort by importance
        order = np.argsort(eigenvalues)[::-1]

        sorted_eigenvalues = eigenvalues[order]
        sorted_eigenvectors = eigenvectors[:, order]

        # Keep only the first k components
        self.components = sorted_eigenvectors[:, :self.n_components]

        # Explained variance
        total_variance = np.sum(sorted_eigenvalues)
        self.explained_variance = sorted_eigenvalues / total_variance

        return self

    def _transform(self, dataset):
        """
        Transform data to the reduced dimensional space using fitted principal components.
        
        Projects the data onto the principal components by:
        1. Centering the data using the training mean
        2. Multiplying by the principal components matrix
        
        Parameters
        ----------
        X : np.ndarray
            Data to transform, shape (n_samples, n_features)
        
        Returns
        -------
        np.ndarray
            Transformed data in reduced dimensional space, shape (n_samples, n_components)
        """
        X = dataset.X 
        X_centered = (X - self.mean) 
        reduced_data = np.dot(X_centered, self.components)


        return reduced_data
    