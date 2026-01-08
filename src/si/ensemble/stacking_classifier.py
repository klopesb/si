import numpy as np
from si.base.model import Model


class StackingClassifier(Model):
    """
    Stacking Classifier ensemble model.
    
    The StackingClassifier uses an ensemble of base models to generate predictions.
    These predictions are then used as features to train a final model that makes
    the ultimate predictions.
    
    Parameters
    ----------
    models : list
        List of base models to use in the ensemble
    final_model : Model
        The model to make the final predictions based on base model outputs
        
    Attributes
    ----------
    models : list
        The fitted base models
    final_model : Model
        The fitted final model
    """
    
    def __init__(self, models, final_model, **kwargs):
        """
        Initialize the StackingClassifier.
        
        Parameters
        ----------
        models : list
            Initial set of base models
        final_model : Model
            The model to make the final predictions
        """
        super().__init__(**kwargs)
        self.models = models
        self.final_model = final_model
        
    def _fit(self, X, y):
        """
        Train the stacking classifier.
        
        Algorithm:
        1. Train the initial set of models
        2. Get predictions from the initial set of models
        3. Train the final model with the predictions of the initial set of models
        4. Return itself (self)
        
        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features)
        y : np.ndarray
            Target values of shape (n_samples,)
            
        Returns
        -------
        self : StackingClassifier
            The fitted model
        """
        # Step 1: Train the initial set of models
        for model in self.models:
            model.fit(X, y)
        
        # Step 2: Get predictions from the initial set of models
        base_predictions = np.column_stack([
            model.predict(X) for model in self.models
        ])
        
        # Step 3: Train the final model with the predictions of the initial set of models
        self.final_model.fit(base_predictions, y)
        
        # Step 4: Return itself
        return self
    
    def _predict(self, X):
        """
        Predict class labels using the stacking classifier.
        
        Algorithm:
        1. Get predictions from the initial set of models
        2. Get the final predictions using the final model and the 
           predictions of the initial set of models
        
        Parameters
        ----------
        X : np.ndarray
            Test data of shape (n_samples, n_features)
            
        Returns
        -------
        predictions : np.ndarray
            Predicted class labels of shape (n_samples,)
        """
        # Step 1: Get predictions from the initial set of models
        base_predictions = np.column_stack([
            model.predict(X) for model in self.models
        ])
        
        # Step 2: Get the final predictions using the final model
        final_predictions = self.final_model.predict(base_predictions)
        
        return final_predictions
    
    def _score(self, X, y):
        """
        Compute the accuracy score of the stacking classifier.
        
        Algorithm:
        1. Get predictions using the predict method
        2. Compute the accuracy between predicted and real values
        
        Parameters
        ----------
        X : np.ndarray
            Test data of shape (n_samples, n_features)
        y : np.ndarray
            True labels of shape (n_samples,)
            
        Returns
        -------
        accuracy : float
            The accuracy score (proportion of correct predictions)
        """
        # Step 1: Get predictions using the predict method
        predictions = self.predict(X)
        
        # Step 2: Compute the accuracy between predicted and real values
        accuracy = np.sum(predictions == y) / len(y)
        
        return accuracy

