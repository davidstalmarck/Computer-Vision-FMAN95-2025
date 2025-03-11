import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def main():
    # 1. Read the images
    im1 = cv2.imread('cube1.jpg')
    im2 = cv2.imread('cube2.jpg')
    
    if im1 is None or im2 is None:
        print("Error: Could not load images. Check file paths.")
        return
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
    # 2. Create the SIFT detector and detect keypoints + descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    print(f"Number of keypoints in image1: {len(kp1)}")
    print(f"Number of keypoints in image2: {len(kp2)}")
    
    # 3. Match the descriptors
    # We use BFMatcher with k-NN matching, then apply Lowe's ratio test
    bf = cv2.BFMatcher(cv2.NORM_L2)
    # k-NN match to get two nearest matches for each descriptor
    matches_knn = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test (Lowe's test) to filter good matches
    ratio_thresh = 0.7
    good_matches = []
    for m, n in matches_knn:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
 
    print(f"Number of good matches after ratio test: {len(good_matches)}")
    
    # 4. Visualization of matches
    # -- Option A: Use OpenCV's built-in function to draw matches
    # This draws lines for *all* the good matches
    matched_img = cv2.drawMatches(im1, kp1, im2, kp2, good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # Show with matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.title('All Good Matches')
    plt.axis('off')
    plt.show()
    exit()

    # -- Option B: Draw only 10 random matches to replicate the “random lines” approach
    # Pick 10 random good matches
    if len(good_matches) > 10:
        sample_matches = random.sample(good_matches, 10)
    else:
        sample_matches = good_matches
 
    matched_img_10 = cv2.drawMatches(im1, kp1, im2, kp2, sample_matches, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                                     matchColor=(0, 255, 0), singlePointColor=None, matchesThickness=5)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(matched_img_10, cv2.COLOR_BGR2RGB))
    plt.title('10 Random Good Matches')
    plt.axis('off')
    plt.show()
    
    # 5. Extract matched (x,y)-coordinates for each image
    #    This is the equivalent of x1, x2 in the assignment
    points1 = []
    points2 = []
    for match in good_matches:
        # QueryIdx is the "index" of the keypoint in the first image
        # TrainIdx is the "index" of the keypoint in the second image
        pt1 = kp1[match.queryIdx].pt  # (x, y) in image1
        pt2 = kp2[match.trainIdx].pt  # (x, y) in image2
        points1.append(pt1)
        points2.append(pt2)
    
    # Convert to NumPy arrays, shape (2, N) if you like
    x1 = np.array(points1)  # shape (2, num_matches)
    x2 = np.array(points2)  # shape (2, num_matches)

    
    print(f"x1 shape: {x1.shape}")
    print(f"x2 shape: {x2.shape}")
    
    # From here, x1 and x2 can be used in your triangulation code.
    return x1, x2
if __name__ == '__main__':
    x1, x2 = main()
    # Pick 10 random samples from x1 and x2
  
    print(x1.tolist())
    print(x2.tolist())

  