import numpy as np
from scipy.linalg import rq
import scipy.io

def normalize_matrix(K):
    """Normalize K matrix by dividing by K[3,3] (MATLAB style indexing in problem)"""
    return K / K[2,2]  # Note: Using Python's 0-based indexing

# Define the transformations from Exercise 1
T1 = np.array([[1, 0, 0, 0],
               [0, 4, 0, 0],
               [0, 0, 1, 0],
               [1/10, 1/10, 0, 1]])

T2 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [1/16, 1/16, 0, 1]])

# Load the data and get the second camera matrix
data = scipy.io.loadmat('compEx1data.mat')
P = data['P'][0]
P2_original = P[1]  # Get second camera (index 1 for second camera)

# Transform the camera matrices
T1_inv = np.linalg.inv(T1)
T2_inv = np.linalg.inv(T2)
P2_T1 = P2_original @ T1_inv
P2_T2 = P2_original @ T2_inv

# Compute RQ decomposition for both P2 matrices
K1, R1 = rq(P2_T1[:3, :3])
K2, R2 = rq(P2_T2[:3, :3])

# Normalize both K matrices
K1_normalized = normalize_matrix(K1)
K2_normalized = normalize_matrix(K2)

# Round the K matrices to 1 decimal place
K1_rounded = np.round(K1_normalized, decimals=1)
K2_rounded = np.round(K2_normalized, decimals=1)

# Set numpy print options for better readability
np.set_printoptions(precision=1, suppress=True)

# Format output for the report with 1 decimal precision
print("For the report:")
print("\nK matrix from P2 with T1 (normalized and rounded):")
print(K1_rounded)
print("\nK matrix from P2 with T2 (normalized and rounded):")
print(K2_rounded)

# Calculate the difference between normalized K matrices
diff = np.abs(K1_rounded - K2_rounded)
print("\nAbsolute difference between normalized K matrices:")
print(np.round(diff, decimals=1))
