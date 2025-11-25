import numpy as np
from si.base.transformer import Transformer

#KB - Exercise 5 

class PCA(Transformer):

    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components
        
        # parâmetros estimados
        self.mean = None
        self.components = None   # eigenvectors
        self.explained_variance = None  # eigenvalues / total

    def _fit(self, X):
        """
        X: matriz de dados (samples x features)
        """

        # ---- Step 1: Center the data ----
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # ---- Step 2: Covariance Matrix ----
        covariance_matrix = np.cov(X_centered, rowvar=False, ddof=1)

        # ---- Step 3: Eigen Decomposition ----
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # ---- Step 4: Sort by importance ----
        order = np.argsort(eigenvalues)[::-1]

        sorted_eigenvalues = eigenvalues[order]
        sorted_eigenvectors = eigenvectors[:, order]

        # guardar apenas os primeiros k componentes
        self.components = sorted_eigenvectors[:, :self.n_components]

        # ---- Step 5: explained variance ----
        total_variance = np.sum(sorted_eigenvalues)
        self.explained_variance = sorted_eigenvalues / total_variance

        return self

    def _transform(self, X):
        """
        Reduz o dataset usando os componentes principais.
        """

        # standardizar com a média aprendida
        X_centered = (X - self.mean) 

        # ---- Step 6: dimensionality reduction ----
        reduced_data = np.dot(X_centered, self.components)


        return reduced_data
    

if __name__ == "__main__":

    X = np.array([
    [   1,   2,  -1,   4,  10],
    [   3,  -3,  -3,  12, -15],
    [   2,   1,  -2,   4,   5],
    [   5,   1,  -5,  10,   5],
    [   2,   3,  -3,   5,  12],
    [   4,   0,  -3,  16,   2],
])

    # ---- Define PCA ----
    pca = PCA(n_components=1)

    # ---- Fit + Transform ----
    X_reduced = pca.fit_transform(X)

    # ---- Output ----
    print("Mean:")
    print(pca.mean)

    print("\nComponents (eigenvectors):")
    print(pca.components)

    print("\nExplained variance (ratio):")
    print(pca.explained_variance)

    print("\nReduced dataset:")
    print(X_reduced)