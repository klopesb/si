import numpy as np

#KB Exercise 4

def tanimoto_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the Tanimoto similarity between a single binary sample and multiple binary samples.
    
    The Tanimoto coefficient is calculated as:
    Similarity = (x · y) / (|x|² + |y|² - x · y)
    
    Parameters
    ----------
    x : np.ndarray
        A single binary sample (1D array)
    y : np.ndarray
        Multiple binary samples (2D array), where each row is a sample
    
    Returns
    -------
    np.ndarray
        An array containing the Tanimoto similarities between x and each sample in y
    """
    
    # x · y (calculate x*y for each row in y)
    dot_product = np.dot(y, x)
    
    # |x|²
    norm_x_squared = np.dot(x, x)
    
    # |y|² (for each row in y)
    norm_y_squared = np.sum(y * y, axis=1)
    
    # Similarity = (x · y) / (|x|² + |y|² - x · y)
    similarity = dot_product / (norm_x_squared + norm_y_squared - dot_product)
    
    return similarity


if __name__ == '__main__':
    # Test 1: Identical vectors
    print("Test 1: Identical vector")
    x1 = np.array([1, 0, 1, 1])
    y1 = np.array([[1, 0, 1, 1]])
    print(f"x = {x1}")
    print(f"y = {y1}")
    print(f"Similarity = {tanimoto_similarity(x1, y1)}")
    print()
    
    # Test 2: Multiple samples
    print("Test 2: Multiple samples")
    x2 = np.array([1, 0, 1, 1])
    y2 = np.array([[1, 0, 1, 1],
                   [1, 1, 0, 0],
                   [0, 1, 0, 0]])
    print(f"x = {x2}")
    print(f"y =\n{y2}")
    print(f"Similarities = {tanimoto_similarity(x2, y2)}")
    print()
    
    # Test 3: Binary example
    print("Test 3: Binary example")
    x3 = np.array([1, 1, 0, 1, 0])
    y3 = np.array([[1, 0, 0, 1, 1],
                   [1, 1, 0, 1, 0],
                   [0, 0, 1, 0, 1]])
    print(f"x = {x3}")
    print(f"y =\n{y3}")
    print(f"Similarities = {tanimoto_similarity(x3, y3)}")