import numpy as np
import matplotlib.pyplot as plt

def psphere(X):
    """
    Normalizes each column vector of X to unit length.
    X is 3 x N (or possibly 2 x N, but here we assume 3 x N).
    Returns the normalized array of the same shape.
    """
    # Compute L2-norm for each column
    norms = np.linalg.norm(X, axis=0)
    # Avoid dividing by zero:
    norms[norms < 1e-15] = 1e-15
    return X / norms

def pflat(X):
    """
    Projects 3D homogeneous points X (3 x N) to 2D by dividing each column by its last coordinate.
    X is typically 3 x N: [X; Y; W].
    Returns a 2 x N array of inhomogeneous points: [X/W; Y/W].
    """
    # Avoid dividing by zero if W=0
    W = X[2, :]
    W[abs(W) < 1e-15] = 1e-15
    return X[:2, :] / W

def rital(linjer, st='-', SCALAR = 1000):
    """
    Python version of rital.m.
    
    linjer: 3 x N numpy array where each column is [a; b; c] 
            representing a line a*x + b*y + c = 0 in homogeneous coords.
    st:     the style string for matplotlib (defaults to '-').

    Draws each line on the current matplotlib figure.
    """
    if linjer.size == 0:
        return  # nothing to plot
    
    # Number of lines N
    _, N = linjer.shape
    
    # rikt = psphere( [b; -a; 0] ) in MATLAB
    # i.e. for each column (a, b, c), we form [b; -a; 0] and then normalize
    a = linjer[0, :]
    b = linjer[1, :]
    # Create [b; -a; 0] for each column
    rikt = np.vstack([b, -a, np.zeros(N)])
    rikt = psphere(rikt)  # normalize each column to length 1
    
    # punkter = pflat( cross(rikt, linjer) )
    # cross(...) in NumPy expects shape (3,) or (N,3), so we do columnwise cross
    # We'll compute cross product for each column i: cross(rikt[:,i], linjer[:,i]).
    # Let's do it in a vectorized way:
    # cross([x1,y1,z1],[x2,y2,z2]) = [y1*z2 - z1*y2, z1*x2 - x1*z2, x1*y2 - y1*x2]
    
    cross_vals = []
    for i in range(N):
        # rikt[:, i] = [rx, ry, rz], linjer[:, i] = [a, b, c]
        rv = rikt[:, i]
        lv = linjer[:, i]
        cr = np.array([
            rv[1]*lv[2] - rv[2]*lv[1],
            rv[2]*lv[0] - rv[0]*lv[2],
            rv[0]*lv[1] - rv[1]*lv[0]
        ])
        cross_vals.append(cr)
    cross_vals = np.array(cross_vals).T  # shape (3, N)
    
    punkter = pflat(cross_vals)
    
    # Finally, plot lines by drawing a segment around each 'punkter' in direction 'rikt'
    # the MATLAB code does a +/- SCALAR factor, so we replicate that:
   
    for i in range(N):
        px, py = punkter[0, i], punkter[1, i]
        rx, ry = rikt[0, i], rikt[1, i]
        x_vals = [px - SCALAR*rx, px + SCALAR*rx]
        y_vals = [py - SCALAR*ry, py + SCALAR*ry]
        plt.plot(x_vals, y_vals, st)
