import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ===============================
# Step 1: Feature Detection and Matching Using SIFT
# ===============================
def detect_and_match_features(img1, img2):
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Feature Matching using FLANN matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Stricter Lowe's Ratio Test
    good_matches = [m for m, n in matches if m.distance < 0.5 * n.distance]

    points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    print(f'Found {len(good_matches)} strong matches.')
    return points1, points2

# ===============================
# Step 2: Essential Matrix Estimation
# ===============================
def estimate_essential_matrix(x1, x2, K, threshold=1.0):
    E, mask = cv2.findEssentialMat(x1, x2, K, method=cv2.RANSAC, threshold=threshold)
    inliers = np.where(mask.ravel() == 1)[0]
    print(f"Estimated Essential Matrix with {len(inliers)} inliers.")
    return E, inliers

# ===============================
# Step 3: Decompose Essential Matrix
# ===============================
def decompose_essential_matrix(E):
    R1, R2, t = cv2.decomposeEssentialMat(E)
    return R1, R2, t

# ===============================
# Step 4: Triangulation
# ===============================
def triangulate_points(P1, P2, x1, x2):
    x1_h = cv2.convertPointsToHomogeneous(x1).reshape(-1, 3).T
    x2_h = cv2.convertPointsToHomogeneous(x2).reshape(-1, 3).T

    points_4d = cv2.triangulatePoints(P1, P2, x1_h[:2], x2_h[:2])
    points_3d = points_4d[:3] / points_4d[3]

    print(f"Triangulated {points_3d.shape[1]} points.")
    return points_3d

# ===============================
# Step 5: Enhanced Outlier Removal
# ===============================
def remove_outliers(U, threshold=2.0):
    mean_position = np.mean(U, axis=1, keepdims=True)
    distances = np.linalg.norm(U - mean_position, axis=0)
    threshold_value = np.mean(distances) + threshold * np.std(distances)
    filtered_U = U[:, distances < threshold_value]

    num_removed = U.shape[1] - filtered_U.shape[1]
    print(f'Removed {num_removed} outliers.')
    return filtered_U

# ===============================
# Step 6: Visualization - Single Step
# ===============================
def visualize_sfm(U, camera_positions, step_title, save_path):
    U_clean = remove_outliers(U)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(U_clean[0], U_clean[1], U_clean[2], c='blue', marker='o', s=5, label='3D Points')
    valid_camera_positions = [pos for pos in camera_positions if len(pos) == 3]
    ax.scatter(*zip(*valid_camera_positions), c='green', marker='^', s=50, label='Camera Positions')

    ax.set_title(f'{step_title}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.savefig(save_path)
    plt.close()

# ===============================
# Step 7: Visualization - Multi-Plot Grid
# ===============================
def visualize_grid(step_images):
    fig, axes = plt.subplots(3, 4, figsize=(12, 10))
    fig.suptitle('Incremental 3D Reconstruction Steps', fontsize=16)

    for idx, ax in enumerate(axes.flat):
        if idx < len(step_images):
            img = plt.imread(step_images[idx])
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Step {idx + 1}')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('Final_MultiPlot_Summary.png')
    plt.show()

# ===============================
# Step 8: Main Pipeline with Image Saving
# ===============================
def incremental_sfm(K, image_files):
    images = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in image_files]
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    camera_positions = [np.array([0, 0, 0])]

    x1, x2 = detect_and_match_features(images[0], images[1])
    E, inliers = estimate_essential_matrix(x1, x2, K)
    R, _, t = decompose_essential_matrix(E)
    P2 = K @ np.hstack((R, t))

    U = triangulate_points(P1, P2, x1[inliers], x2[inliers])
    total_points = U.shape[1]

    step_images = []
    # Visualization per step
    step_image_path = f'step_{1}.png'
    visualize_sfm(U, camera_positions, f"Step {1}", step_image_path)
    step_images.append(step_image_path)
    for i in range(1, len(images) - 1): 
        print(f"\n=== Registering Image {i} and {i+1} ===")
        x1_new, x2_new = detect_and_match_features(images[i], images[i + 1])
        E, inliers = estimate_essential_matrix(x1_new, x2_new, K)
        R_new, _, t_new = decompose_essential_matrix(E)
        P_new = K @ np.hstack((R_new, t_new))
        U_new = triangulate_points(P1, P_new, x1_new[inliers], x2_new[inliers])

        # Combine points
        U = np.hstack((U, U_new))
        camera_position = -R_new.T @ t_new
        camera_positions.append(camera_position.flatten())

        # Visualization per step
        step_image_path = f'step_{i+1}.png'
        visualize_sfm(U, camera_positions, f"Step {i+1}", step_image_path)
        step_images.append(step_image_path)

        total_points_after_step = U.shape[1]
        new_points_added = total_points_after_step - total_points
        total_points = total_points_after_step

        print(f'Points added in Step {i} → {new_points_added} new points.')

    print("3D Reconstruction Complete")
    visualize_grid(step_images)

# ===============================
# Step 9: Load Data and Run Pipeline
# ===============================
if __name__ == "__main__":
    from scipy.io import loadmat

    # Load K from compEx5data.mat
    data = loadmat('compEx5data.mat')
    K = data['K']

    # Load image file paths
    image_files = sorted(glob.glob('house/*.jpg'))

    incremental_sfm(K, image_files)
