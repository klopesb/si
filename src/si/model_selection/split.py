from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

#KB Exercise 6.1
def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets using stratified sampling.
    This ensures that the proportion of samples for each class is preserved in both sets.

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    
    # Get unique class labels and their counts
    unique_labels = np.unique(dataset.y)
    
    # Initialize empty lists for train and test indices
    train_idxs = []
    test_idxs = []
    
    # Loop through unique labels
    for label in unique_labels:
        # Get indices for the current class
        label_idxs = np.where(dataset.y == label)[0]
        
        # Calculate the number of test samples for the current class
        n_test_class = int(len(label_idxs) * test_size)
        
        # Shuffle indices for the current class
        np.random.shuffle(label_idxs)
        
        # Select indices for test set
        test_idxs_class = label_idxs[:n_test_class]
        
        # Select remaining indices for train set
        train_idxs_class = label_idxs[n_test_class:]
        
        # Add to the main lists
        test_idxs.extend(test_idxs_class)
        train_idxs.extend(train_idxs_class)
    
    # Convert lists to numpy arrays
    train_idxs = np.array(train_idxs)
    test_idxs = np.array(test_idxs)
    
    # Create training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    
    return train, test


if __name__ == '__main__':
    # Example usage with a synthetic imbalanced dataset
    from si.data.dataset import Dataset
    
    # Create a synthetic dataset with imbalanced classes
    # Class 0: 100 samples, Class 1: 50 samples, Class 2: 30 samples
    np.random.seed(42)
    X = np.random.randn(180, 4)
    y = np.array([0] * 100 + [1] * 50 + [2] * 30)
    
    dataset = Dataset(X=X, y=y, features=["f1", "f2", "f3", "f4"], label="class")
    
    print("=" * 70)
    print("ORIGINAL DATASET")
    print("=" * 70)
    print(f"Total samples: {len(dataset.y)}")
    unique, counts = np.unique(dataset.y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples ({count/len(dataset.y)*100:.1f}%)")
    print()
    
    # Test regular train_test_split
    print("=" * 70)
    print("REGULAR TRAIN_TEST_SPLIT (test_size=0.2)")
    print("=" * 70)
    train_regular, test_regular = train_test_split(dataset, test_size=0.2, random_state=42)
    
    print(f"Train set: {len(train_regular.y)} samples")
    unique, counts = np.unique(train_regular.y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples ({count/len(train_regular.y)*100:.1f}%)")
    
    print(f"\nTest set: {len(test_regular.y)} samples")
    unique, counts = np.unique(test_regular.y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples ({count/len(test_regular.y)*100:.1f}%)")
    print()
    
    # Test stratified_train_test_split
    print("=" * 70)
    print("STRATIFIED_TRAIN_TEST_SPLIT (test_size=0.2)")
    print("=" * 70)
    train_strat, test_strat = stratified_train_test_split(dataset, test_size=0.2, random_state=42)
    
    print(f"Train set: {len(train_strat.y)} samples")
    unique, counts = np.unique(train_strat.y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples ({count/len(train_strat.y)*100:.1f}%)")
    
    print(f"\nTest set: {len(test_strat.y)} samples")
    unique, counts = np.unique(test_strat.y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples ({count/len(test_strat.y)*100:.1f}%)")
    print()
    
    # Compare proportions
    print("=" * 70)
    print("COMPARISON: Class proportions preserved?")
    print("=" * 70)
    
    original_props = np.bincount(dataset.y) / len(dataset.y)
    train_strat_props = np.bincount(train_strat.y) / len(train_strat.y)
    test_strat_props = np.bincount(test_strat.y) / len(test_strat.y)
    
    print(f"{'Class':<10} {'Original':<15} {'Train (Strat)':<15} {'Test (Strat)':<15}")
    print("-" * 70)
    for i in range(len(original_props)):
        print(f"{i:<10} {original_props[i]*100:>6.1f}%{'':<8} "
              f"{train_strat_props[i]*100:>6.1f}%{'':<8} "
              f"{test_strat_props[i]*100:>6.1f}%{'':<8}")
    print()
    print("✓ Stratified split preserves class proportions in both train and test sets!")