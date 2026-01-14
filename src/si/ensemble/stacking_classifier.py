import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


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
    
    def __init__(self, models: list, final_model: Model, **kwargs):
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
        
    def _fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Train the stacking classifier.
        
        Algorithm:
        1. Train the initial set of models
        2. Get predictions from the initial set of models
        3. Train the final model with the predictions of the initial set of models
        4. Return itself (self)
        
        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the model to
            
        Returns
        -------
        self : StackingClassifier
            The fitted model
        """
        # Step 1: Train the initial set of models
        for model in self.models:
            model.fit(dataset)
        
        # Step 2: Get predictions from the initial set of models
        base_predictions = np.column_stack([
            model.predict(dataset) for model in self.models
        ])
        
        # Step 3: Train the final model with the predictions of the initial set of models
        # Create a new dataset with base predictions as features and original labels
        stacked_dataset = Dataset(X=base_predictions, y=dataset.y)
        self.final_model.fit(stacked_dataset)
        
        # Step 4: Return itself
        return self
    
    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class labels using the stacking classifier.
        
        Algorithm:
        1. Get predictions from the initial set of models
        2. Get the final predictions using the final model and the 
           predictions of the initial set of models
        
        Parameters
        ----------
        dataset : Dataset
            The dataset to predict the classes of
            
        Returns
        -------
        predictions : np.ndarray
            Predicted class labels
        """
        # Step 1: Get predictions from the initial set of models
        base_predictions = np.column_stack([
            model.predict(dataset) for model in self.models
        ])
        
        # Step 2: Get the final predictions using the final model
        # Create a new dataset with base predictions as features
        stacked_dataset = Dataset(X=base_predictions, y=dataset.y)
        final_predictions = self.final_model.predict(stacked_dataset)
        
        return final_predictions
    
    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Compute the accuracy score of the stacking classifier.
        
        Algorithm:
        1. Compute the accuracy between predicted and real values
        
        Parameters
        ----------
        dataset : Dataset
            The dataset to evaluate the model on
        predictions : np.ndarray
            An array with the predictions
            
        Returns
        -------
        accuracy : float
            The accuracy score (proportion of correct predictions)
        """
        # Compute the accuracy between predicted and real values
        return accuracy(dataset.y, predictions)
