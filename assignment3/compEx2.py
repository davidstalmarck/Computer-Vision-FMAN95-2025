import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.linalg import svd, norm, solve
from scipy.linalg import null_space
import cv2
########################################
# 1. LOAD DATA
########################################

data = loadmat('compEx1data.mat')
x = data['x']  


x1 = x[0,0]  # first set of points, shape is 3×N
x2 = x[1,0]  # second set of points, shape is 3×N

F =np.array([[-3.3901069316072084e-08, -3.7200533844642334e-06, 0.005772312832305452], [4.667369031726206e-06, 2.893608325990422e-07, -0.026682103359356633], [-0.007193603737897985, 0.026295710857867208, 1.0]])


########################################
# 2. BUILD CAMERA MATRICES
########################################

# Camera 1 is [I | 0]
P1 = np.hstack((np.eye(3), np.zeros((3,1))))  # shape (3, 4)

# Compute epipole e2 (null space of F^T)
# e2 is the vector in the right null space of F^T
#   i.e. F^T e2 = 0

'''
ns = null_space(F.T)  # shape (3, k)
# We should get exactly one basis vector (k=1 for rank-2 F)
e2 = ns[:, 0]  # The single null-space vector
'''

u, s, vh = np.linalg.svd(F.T)
# The last column of vh^T (or equivalently the last row of vh) is the null vector
e2 = vh[-1, :]
# Make homogeneous
e2 = e2/e2[-1]  # ensure last coordinate is 1 if possible

# Build [e2]_x
def cross_matrix(v):
    """ Return the skew-symmetric cross-product matrix [v]_x. """
    return np.array([
        [     0, -v[2],  v[1]],
        [  v[2],     0, -v[0]],
        [ -v[1],  v[0],     0]
    ])

e2x = cross_matrix(e2)

# Now build the second camera matrix
# A = [e2]_x * F
A = e2x @ F
P2 = np.hstack((A, e2.reshape(3,1)))  # shape (3, 4)


# 
########################################
# 3. TRIANGULATE 3D POINTS
########################################

def triangulate_dlt(P1, P2, x1, x2):
    """ Perform Direct Linear Transform (DLT) triangulation """
    num_points = x1.shape[1]
    X = np.zeros((4, num_points))

    for i in range(num_points):
        A = np.vstack([
            x1[0, i] * P1[2, :] - P1[0, :],
            x1[1, i] * P1[2, :] - P1[1, :],
            x2[0, i] * P2[2, :] - P2[0, :],
            x2[1, i] * P2[2, :] - P2[1, :]
        ])
        _, _, V = svd(A)
        X[:, i] = V[-1]  # Last row of V gives the solution

    X /= X[3, :]  # Convert to inhomogeneous coordinates
    return X[:3, :]

# Triangulate 3D points
X_3D = triangulate_dlt(P1, P2, x1, x2)
def plot_2d_with_image(image, x_true, x_proj, title="2D Projection"):
    """ Plot the original 2D points and projected 3D points on the image """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)  # Display the actual image
    ax.scatter(x_true[0, :], x_true[1, :], c='r', marker='o', label="Measured 2D Points")
    ax.scatter(x_proj[0, :], x_proj[1, :], marker='x', c='g', label="Projected 3D Points")
    ax.set_title(title)
    ax.legend()
    plt.show()

# Load images
img1 = plt.imread("kronan1.jpg")  # Load first image
img2 = plt.imread("kronan2.jpg")  # Load second image

########################################
# 4. RE-PROJECT POINTS AND VISUALIZE (SUBPLOTS)
########################################

# Re-project each 3D point into the two cameras
x1_rep = P1 @ np.vstack((X_3D, np.ones((1, X_3D.shape[1]))))  # shape (3, N)
x2_rep = P2 @ np.vstack((X_3D, np.ones((1, X_3D.shape[1]))))  # shape (3, N)
x1_rep /= x1_rep[2, :]
x2_rep /= x2_rep[2, :]

# Create a subplot to show both images side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot Image 1 with measured and reprojected points
axes[0].imshow(img1)
axes[0].scatter(x1[0, :], x1[1, :], c='r', marker='o', label='Measured x1', s=10)
axes[0].scatter(x1_rep[0, :], x1_rep[1, :], c='g', marker='x', label='Reproj x1', s=10)
axes[0].set_title("Measured vs. Reprojected in Image 1")
axes[0].legend()

# Plot Image 2 with measured and reprojected points
axes[1].imshow(img2)
axes[1].scatter(x2[0, :], x2[1, :], c='r', marker='o', label='Measured x2', s=10)
axes[1].scatter(x2_rep[0, :], x2_rep[1, :], c='g', marker='x', label='Reproj x2', s=10)
axes[1].set_title("Measured vs. Reprojected in Image 2")
axes[1].legend()

# Display the subplots
plt.show()


########################################
# 5. 3D PLOT
########################################

# Optional: Plot the 3D points + camera centers
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')

# Extract camera centers
# For P1 = [I|0], center is (0,0,0)
C1 = np.array([0,0,0,1])
# For P2 = [A|e2], the center solves A*C2 + e2 = 0 => C2 = -inv(A)*e2
# If A is rank-deficient, there is an alternative approach. 
#   But let's do naive version if A is invertible:

# Compute second camera center C2 by solving A * C2 = -c2
A = P2[:, :3]  # Extract 3x3 rotation part of P2
c2 = P2[:, 3]  # Extract translation vector
# Solve the equation A * C2 = -c2
if np.linalg.matrix_rank(A) == 3:  # Ensure A is full-rank
    C2 = np.append(solve(A, -c2), 1.0)  # Convert to homogeneous coordinates
else:
    print("Matrix A is singular, camera center at infinity.")
    C2 = np.array([np.nan, np.nan, np.nan, np.nan])

# 3D points:
X_inh = X_3D[:3,:]  # inhomogeneous coords (3 x N)

# Plot them
ax2.scatter(X_inh[0,:], X_inh[1,:], X_inh[2,:], c='b', marker='o', s=5, label='3D points')
# Plot camera centers
ax2.scatter(C1[0], C1[1], C1[2], c='r', marker='^', s=50, label='Camera 1')
if not np.any(np.isnan(C2)):
    ax2.scatter(C2[0], C2[1], C2[2], c='g', marker='^', s=50, label='Camera 2')

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('3D Points and Cameras')
ax2.legend()
plt.show()
