import numpy as np
import matplotlib.pyplot as plt

def pflat(X):
    """
    Normalize homogeneous coordinates by dividing by the last coordinate.
    """
    return X / X[-1]

def triangulate_point(P1, x1, P2, x2):
    """
    Triangulate a single 3D point from two views using DLT.
    P1, P2: 3x4 camera matrices
    x1, x2: 3D homogeneous image points (3,)
    Returns X: 4x1 homogeneous 3D point
    """
    p1_1, p1_2, p1_3 = P1[0], P1[1], P1[2]
    p2_1, p2_2, p2_3 = P2[0], P2[1], P2[2]

    u1, v1, _ = x1
    u2, v2, _ = x2

    A = np.vstack([
        u1 * p1_3 - p1_1,
        v1 * p1_3 - p1_2,
        u2 * p2_3 - p2_1,
        v2 * p2_3 - p2_2
    ])

    # Solve A X = 0 using SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]  # Last row of Vt is the solution
    return X / X[-1]  # Normalize to make it non-homogeneous

def triangulate_all(P1, x1_points, P2, x2_points):
    """
    Triangulate multiple 3D points using DLT.
    Returns X_all: 4xN array of 3D points.
    """
    X_all = np.zeros((4, len(x1_points)))

    for i in range(len(x1_points)):
        X_all[:, i] = triangulate_point(P1, x1_points[i], P2, x2_points[i])
    
    return X_all

# Camera matrices
P1 = np.array([[-2391.5, 225.7, 942.7, 0],
               [0.0, 623.9, 813.6, 0],
               [0.0, 0.0, 1.0, 0]])

P2 = np.array([[-2394.0, 0.0, 932.4, 0],
               [0.0, 2398.1, 628.3, 0],
               [0.0, 0.0, 1.0, 0]])

# 10 Matched 2D points in homogeneous coordinates
x1 = np.array([[443.22, 589.06, 1], [1625.93, 1184.23, 1], [314.42, 710.76, 1], 
               [463.81, 634.42, 1], [1349.58, 1076.26, 1], [1684.23, 1178.88, 1], 
               [435.74, 323.95, 1], [1656.00, 946.95, 1], [1114.50, 937.29, 1], 
               [898.70, 1048.14, 1]])

x2 = np.array([[379.47, 515.33, 1], [1538.82, 1195.78, 1], [265.14, 626.49, 1], 
               [401.42, 560.54, 1], [1279.52, 1062.57, 1], [1603.14, 1196.49, 1], 
               [360.62, 257.06, 1], [1648.19, 957.95, 1], [764.76, 813.90, 1], 
               [795.72, 990.79, 1]])

# Triangulate all points
X_3D = triangulate_all(P1, x1, P2, x2)


def project_points(X, P):
    """
    Projects 3D points X into an image using camera matrix P.
    X: 4xN array of homogeneous 3D points
    P: 3x4 camera projection matrix
    Returns projected 2D points (inhomogeneous coordinates).
    """
    x_proj = P @ X  # Project 3D points using P
    return pflat(x_proj)  # Convert to inhomogeneous coordinates

# Project the 3D points into both images
x1_proj = project_points(X_3D, P1)
x2_proj = project_points(X_3D, P2)

# Plot results for Image 1
plt.figure(figsize=(8,6))
img1 = np.zeros((1200, 1600))  # Dummy image (replace with actual image)
plt.imshow(img1, cmap="gray")

plt.scatter(x1[:, 0], x1[:, 1], c='b', marker='o', label="Original SIFT Points")
plt.scatter(x1_proj[0, :], x1_proj[1, :], c='r', marker='x', label="Projected 3D Points")
plt.legend()
plt.title("Projection in Image 1")
plt.show()

# Plot results for Image 2
plt.figure(figsize=(8,6))
img2 = np.zeros((1200, 1600))  # Dummy image (replace with actual image)
plt.imshow(img2, cmap="gray")

plt.scatter(x2[:, 0], x2[:, 1], c='b', marker='o', label="Original SIFT Points")
plt.scatter(x2_proj[0, :], x2_proj[1, :], c='r', marker='x', label="Projected 3D Points")
plt.legend()
plt.title("Projection in Image 2")
plt.show()
