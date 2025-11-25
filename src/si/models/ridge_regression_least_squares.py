#KB Exercise 8

import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares(Model):
    """
        parameters:
    - l2_penalty -  L2 regularization parameter
    - scale - wheter to scale the data or not
    • estimated parameters:
    - theta - the coefficients of the model for every feature
    - theta_zero - the zero coefficient (y intercept)
    - mean - mean of the dataset (for every feature)
    - std - standard deviation of the dataset (for every feature)
    • methods:
    - fit - estimates the theta and theta_zero coefficients, mean and std
    - predict - predicts the dependent variable (y) using the estimated theta coefficients
    - score - calculates the error between the real and predicted y values 
    """
    def __init__(self, l2_penalty: float = 1.0, scale: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.scale = scale
        
        # estimated parameters
        self.theta = None          # coefficients
        self.theta_zero = None     # intercept
        self.mean = None           # feature means
        self.std = None            # feature std
        

    def _fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
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
        # 1. mse
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