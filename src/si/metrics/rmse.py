import numpy as np
from si.metrics.mse import mse

#KB exercise 7.1

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted values.
    
    RMSE = sqrt(1/n * sum((y_pred_i - y_true_i)^2))
    
    Parameters
    ----------
    y_true : np.ndarray
        Real values of y (ground truth)
    y_pred : np.ndarray
        Predicted values of y
    
    Returns
    -------
    float
        RMSE error between y_true and y_pred
    
    Examples
    --------
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    >>> rmse(y_true, y_pred)
    0.612...
    """
    
    # Calculate square root (RMSE)
    return np.sqrt(mse(y_true, y_pred))



if __name__ == '__main__':
    print("=" * 70)
    print("TEST 1: Perfect predictions")
    print("=" * 70)
    y_true1 = np.array([1, 2, 3, 4, 5])
    y_pred1 = np.array([1, 2, 3, 4, 5])
    print(f"y_true: {y_true1}")
    print(f"y_pred: {y_pred1}")
    print(f"RMSE: {rmse(y_true1, y_pred1):.6f}")
    print("Expected: 0.0 (perfect predictions)")
    print()
    
    print("=" * 70)
    print("TEST 2: Small errors")
    print("=" * 70)
    y_true2 = np.array([3, -0.5, 2, 7])
    y_pred2 = np.array([2.5, 0.0, 2, 8])
    print(f"y_true: {y_true2}")
    print(f"y_pred: {y_pred2}")
    print(f"RMSE: {rmse(y_true2, y_pred2):.6f}")
    print("Errors: [-0.5, 0.5, 0, 1]")
    print()
    
    print("=" * 70)
    print("TEST 3: Larger errors")
    print("=" * 70)
    y_true3 = np.array([10, 20, 30, 40, 50])
    y_pred3 = np.array([12, 18, 35, 38, 55])
    print(f"y_true: {y_true3}")
    print(f"y_pred: {y_pred3}")
    print(f"RMSE: {rmse(y_true3, y_pred3):.6f}")
    print("Errors: [2, -2, 5, -2, 5]")
    print()
    
    print("=" * 70)
    print("TEST 5: Comparison - small vs large errors")
    print("=" * 70)
    y_true5 = np.array([10, 10, 10, 10])
    
    # Case A: consistent small errors
    y_pred5a = np.array([11, 11, 11, 11])
    rmse_a = rmse(y_true5, y_pred5a)
    print(f"Case A - Consistent small errors (+1 each):")
    print(f"  y_pred: {y_pred5a}")
    print(f"  RMSE: {rmse_a:.6f}")
    
    # Case B: one large error
    y_pred5b = np.array([10, 10, 10, 14])
    rmse_b = rmse(y_true5, y_pred5b)
    print(f"\nCase B - One large error (+4 on last):")
    print(f"  y_pred: {y_pred5b}")
    print(f"  RMSE: {rmse_b:.6f}")
    
    print(f"\nNote: RMSE penalizes large errors more heavily!")
    print(f"Case A (4 errors of +1): RMSE = {rmse_a:.3f}")
    print(f"Case B (1 error of +4): RMSE = {rmse_b:.3f}")
    print()
    
    print("=" * 70)
    print("TEST 6: Real-world example - House price prediction")
    print("=" * 70)
    # Prices in thousands of dollars
    y_true6 = np.array([250, 300, 350, 400, 450])  # True prices
    y_pred6 = np.array([245, 310, 340, 395, 460])  # Predicted prices
    
    print(f"True prices (K$): {y_true6}")
    print(f"Predicted prices (K$): {y_pred6}")
    print(f"RMSE: ${rmse(y_true6, y_pred6):.2f}K")
    print(f"\nInterpretation: On average, predictions are off by ~${rmse(y_true6, y_pred6):.2f}K")