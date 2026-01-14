from typing import Dict, Callable, List, Any
import numpy as np
import itertools
from si.base.model import Model
from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation



def randomized_search_cv(model: Model,
                         dataset: Dataset,
                         hyperparameter_grid: Dict[str, List[Any]],
                         scoring: Callable = None,
                         cv: int = 3,
                         n_iter: int = 10,
                         seed: int = None) -> Dict[str, Any]:
    """
    Perform randomized search cross-validation on the given model and dataset.
    
    This function randomly samples n_iter combinations from the hyperparameter grid,
    evaluates each combination using k-fold cross-validation, and returns the best
    performing hyperparameters.

    Parameters
    ----------
    model : Model
        The model to validate
    dataset : Dataset
        The validation dataset
    hyperparameter_grid : Dict[str, List[Any]]
        Dictionary with the hyperparameter names as keys and lists of possible
        values to search as values
    scoring : Callable, optional
        Score function to use. If None, the model's score method will be used
    cv : int, default=3
        Number of cross-validation folds
    n_iter : int, default=10
        Number of random hyperparameter combinations to test
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    results : Dict[str, Any]
        Dictionary with the following keys:
        - 'hyperparameters': list of all hyperparameter combinations tested
        - 'scores': list of mean scores obtained for each combination
        - 'best_hyperparameters': best combination of hyperparameters
        - 'best_score': best mean score achieved

    Raises
    ------
    ValueError
        If any hyperparameter in the grid does not exist in the model

    """

    # Step 1: Check if the provided hyperparameters are valid
    for hyperparameter in hyperparameter_grid.keys():
        if not hasattr(model, hyperparameter):
            raise ValueError(f"Model {model.__class__.__name__} does not have "
                           f"hyperparameter '{hyperparameter}'")
    
    # Step 2: Get n_iter hyperparameter combinations
    # Generate all possible combinations
    hyperparameter_names = list(hyperparameter_grid.keys())
    hyperparameter_values = list(hyperparameter_grid.values())
    all_combinations = list(itertools.product(*hyperparameter_values))
    
    # Randomly sample n_iter combinations (or all if n_iter > total combinations)
    if seed is not None:
        np.random.seed(seed)
    
    n_combinations = min(n_iter, len(all_combinations))
    random_indices = np.random.choice(len(all_combinations), 
                                     size=n_combinations, 
                                     replace=False)
    selected_combinations = [all_combinations[i] for i in random_indices]
    
    # Initialize results storage
    all_hyperparameters = []
    all_scores = []
    best_score = -np.inf
    best_hyperparameters = None
    
    # Step 3-6: Iterate through all selected hyperparameter combinations
    for combination in selected_combinations:
        # Step 3: Set the model hyperparameters with the current combination
        current_hyperparameters = {}
        for hyperparameter_name, hyperparameter_value in zip(hyperparameter_names, combination):
            setattr(model, hyperparameter_name, hyperparameter_value)
            current_hyperparameters[hyperparameter_name] = hyperparameter_value
        
        # Step 4: Cross validate the model using k_fold_cross_validation
        fold_scores = k_fold_cross_validation(model, dataset, scoring=scoring, cv=cv, seed=seed)
        
        # Step 5: Save the mean of the scores and respective hyperparameters
        mean_score = np.mean(fold_scores)
        all_hyperparameters.append(current_hyperparameters.copy())
        all_scores.append(mean_score)
        
        # Step 7: Update best score and hyperparameters if current is better
        if mean_score > best_score:
            best_score = mean_score
            best_hyperparameters = current_hyperparameters.copy()
    
    # Step 8: Return results dictionary
    results = {
        'hyperparameters': all_hyperparameters,
        'scores': all_scores,
        'best_hyperparameters': best_hyperparameters,
        'best_score': best_score
    }
    
    return results

