import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat

def pflat(X):
    return X[:-1] / X[-1]

def null(A):
    """
    Return a vector in the null space of A using SVD.
    For a 3x4 camera matrix P, this yields the camera center.
    """
    # SVD: A = U * Sigma * V^T
    # The null space vector is the last row of V^T.
    u, s, vh = np.linalg.svd(A)
    return vh[-1,:].reshape(-1,1)  # shape (4,1) in homogeneous coords

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so the data isn’t distorted.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)
    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)
    ax.set_xlim3d([mid_x - 0.5*max_range, mid_x + 0.5*max_range])
    ax.set_ylim3d([mid_y - 0.5*max_range, mid_y + 0.5*max_range])
    ax.set_zlim3d([mid_z - 0.5*max_range, mid_z + 0.5*max_range])


# --- Main script ---

# 1) Load images and .mat data
im1 = plt.imread('compEx4im1.jpg')
im2 = plt.imread('compEx4im2.jpg')
data = loadmat('compEx4.mat')

K  = data['K']      # Calibration matrix
R1 = data['R1']     # Rotation 1
t1 = data['t1']     # Translation 1
R2 = data['R2']     # Rotation 2
t2 = data['t2']     # Translation 2
U  = data['U']      # 4xN homogeneous points of the statue

# 2) Display the images
fig_images, axs = plt.subplots(1,2, figsize=(10,5))
axs[0].imshow(im1, cmap='gray')
axs[0].set_title('compEx4im1.jpg')
axs[1].imshow(im2, cmap='gray')
axs[1].set_title('compEx4im2.jpg')
for ax in axs:
    ax.axis('off')

# 3) Form the camera matrices: P = K [R | t]
P1 = K @ np.hstack([R1, t1])  # shape (3,4)
P2 = K @ np.hstack([R2, t2])

# 4) Camera centers as the null space of each P
C1_h = null(P1)  # homogeneous 4D
C2_h = null(P2)
C1 = pflat(C1_h) # convert to 3D
C2 = pflat(C2_h)

# 5) Principal axes (the 'viewing direction' is 3rd row of R)
principal_axis1 = R1[2,:]  # shape (1,3)
principal_axis2 = R2[2,:]

# 6) pflat the 3D points
U_3D = pflat(U)  # shape (3, N)

# 7) Plot the 3D points, camera centers, and principal axes
fig_3d = plt.figure(figsize=(8,6))
ax = fig_3d.add_subplot(111, projection='3d')

# Points in U
ax.scatter(U_3D[0,:], U_3D[1,:], U_3D[2,:],
           c='b', s=1, label='Points in U')

# Camera centers
ax.scatter(C1[0], C1[1], C1[2],
           c='r', s=50, marker='o', label='Camera 1 Center')
ax.scatter(C2[0], C2[1], C2[2],
           c='g', s=50, marker='o', label='Camera 2 Center')

# Principal axes as 3D quiver arrows
scale = 100  
ax.quiver(C1[0], C1[1], C1[2],
          principal_axis1[0]*scale,
          principal_axis1[1]*scale,
          principal_axis1[2]*scale,
          color='r', linewidth=2, label='Principal Axis 1')
ax.quiver(C2[0], C2[1], C2[2],
          principal_axis2[0]*scale,
          principal_axis2[1]*scale,
          principal_axis2[2]*scale,
          color='g', linewidth=2, label='Principal Axis 2')

# Make axes equal and label them
set_axes_equal(ax)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Points, Camera Centers, & Principal Axes')
ax.legend(loc='best')

plt.show()
