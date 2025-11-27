#KB Exercise 8

import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares(Model):
    """
    Ridge Regression using the closed-form Least Squares solution with L2 regularization.
    
    Ridge regression adds an L2 penalty term to the ordinary least squares objective,
    which helps prevent overfitting by shrinking the coefficients. The model is solved
    using the closed-form solution: θ = (X^T X + λI)^(-1) X^T y
    
    Parameters
    ----------
    l2_penalty : float, default=1.0
        L2 regularization parameter (lambda). Higher values increase regularization strength
    scale : bool, default=True
        Whether to standardize features by removing mean and scaling to unit variance
    **kwargs : dict
        Additional keyword arguments passed to the parent Model class
    
    Attributes
    ----------
    theta : np.ndarray
        Coefficients for each feature (excluding intercept)
    theta_zero : float
        Intercept term (bias)
    mean : np.ndarray
        Mean of each feature in the training data (used for scaling)
    std : np.ndarray
        Standard deviation of each feature in the training data (used for scaling)
    """

    def __init__(self, l2_penalty: float = 1.0, scale: bool = True, **kwargs):
        """
        Initialize the Ridge Regression model.
        
        Parameters
        ----------
        l2_penalty : float, default=1.0
            L2 regularization strength (lambda)
        scale : bool, default=True
            If True, standardize features before fitting
        **kwargs : dict
            Additional arguments for the parent Model class
        """
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.scale = scale
        
        # estimated parameters
        self.theta = None          # coefficients
        self.theta_zero = None     # intercept
        self.mean = None           # feature means
        self.std = None            # feature std
        

    def _fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Fit the Ridge Regression model using the closed-form solution.
        
        The fitting process:
        1. Optionally standardizes the features
        2. Adds intercept term to the design matrix
        3. Creates penalty matrix (λI) without penalizing the intercept
        4. Solves the normal equation: θ = (X^T X + λI)^(-1) X^T y
        5. Separates intercept from feature coefficients
        
        Parameters
        ----------
        dataset : Dataset
            Training dataset containing features (X) and targets (y)
        
        Returns
        -------
        self : RidgeRegressionLeastSquares
            The fitted model instance
        """
        X = dataset.X
        y = dataset.y
        
        # 1. Scale data if needed
        if self.scale:
            self.mean = np.nanmean(X, axis=0)
            self.std = np.nanstd(X, axis=0)
            X = (X - self.mean) / self.std
        
        m, n = X.shape
        
        # 2. Add intercept term
        X_bias = np.c_[np.ones(m), X]     # shape = (m, n+1)

        # 3. Penalty matrix (lambda * I)
        penalty = self.l2_penalty * np.eye(n + 1)

        # 4. Do NOT penalize intercept
        penalty[0, 0] = 0

        # 5. Compute closed form solution:
        # (X^T X + λI)^(-1) (X^T y)
        XtX = X_bias.T.dot(X_bias)
        XtY = X_bias.T.dot(y)
        
        thetas = np.linalg.inv(XtX + penalty).dot(XtY)

        # Separate theta_zero do resto
        self.theta_zero = thetas[0]
        self.theta = thetas[1:]
        
        return self


    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict target values for new data using the fitted Ridge Regression model.
        
        The prediction process:
        1. Standardizes features using training mean/std (if scaling was enabled)
        2. Adds intercept term to the design matrix
        3. Computes predictions using: y_pred = X θ + θ_0
        
        Parameters
        ----------
        dataset : Dataset
            Dataset containing features (X) to make predictions on
        
        Returns
        -------
        np.ndarray
            Predicted target values, shape (n_samples,)
        """
        X = dataset.X
        
        # 1. Scale using train mean/std
        if self.scale:
            X = (X - self.mean) / self.std
        
        # 2. Add intercept term
        m = X.shape[0]
        X_bias = np.c_[np.ones(m), X]

        # 3. Concatenate intercept + coef.
        thetas = np.r_[self.theta_zero, self.theta]

        # 4. Compute predictions
        return X_bias.dot(thetas)


    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculate the Mean Squared Error (MSE) between predictions and actual values.
        
        Parameters
        ----------
        dataset : Dataset
            Dataset containing true target values (y)
        predictions : np.ndarray
            Predicted target values
        
        Returns
        -------
        float
            Mean Squared Error (MSE) - lower values indicate better fit
        """
        return mse(dataset.y, predictions)
    
if __name__ == '__main__':
# imports necessários
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split

    # criar dataset aleatório
    dataset = Dataset.from_random(n_samples=200, n_features=5)

    # dividir entre treino e teste
    train, test = train_test_split(dataset, test_size=0.2)

    # criar modelo
    model = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)

    # treinar
    model.fit(train)

    # mostrar parâmetros aprendidos
    print("Theta (coeficientes):")
    print(model.theta)

    print("\nTheta_zero (intercepto):")
    print(model.theta_zero)

    # prever no test set
    predictions = model.predict(test)

    print("\nPrimeiras 10 previsões:")
    print(predictions[:10])

    print("\nPrimeiros 10 valores reais:")
    print(test.y[:10])

    # calcular score
    score = model.score(test)
    print(f"\nMSE score no test set: {score:.4f}")