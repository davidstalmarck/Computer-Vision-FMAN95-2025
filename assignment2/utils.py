import numpy as np

def pflat(X):
    """
    Convert homogeneous coordinates X (4 x N or 3 x N) to inhomogeneous
    by dividing by the last row. Returns the same shape as X.
    """
    # If X is shape (d,N), last coordinate is X[d-1, :]
    # Be sure not to divide by zero. Check for X[-1,:] != 0 if needed.
    return X / X[-1, :]

def load_mat(filename):
    """
    Load a .mat file into a Python dictionary. 
    """
    from scipy.io import loadmat
    return loadmat(filename, squeeze_me=True, struct_as_record=False)

def svd_solve_homogeneous(M):
    """
    Solve M v = 0 in a least-squares sense using SVD.
    Returns the 1D array (vector) v that is the last column of V.
    """
    U, S, Vt = np.linalg.svd(M)
    v = Vt[-1, :]  # Last row of V^T => last column of V
    return v

def apply_transform(T, X):
    """
    Apply 4x4 transform T to 4xN array X in homogeneous coords.
    Returns pflat(T @ X).
    """
    return pflat(T @ X)
