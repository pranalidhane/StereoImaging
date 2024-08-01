# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:16:06 2024

@author: prana
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from DetectMarkers import detectMarkers


# Find corresponding point in the rectified image
def find_corresponding_point(point, H):
    # Convert the point to homogeneous coordinates
    point_homogeneous = np.array([point[0], point[1], 1]).reshape(3, 1)

    # Apply the homography to the point
    rectified_point_homogeneous = np.dot(H, point_homogeneous)

    # Convert back to Cartesian coordinates
    rectified_point = (rectified_point_homogeneous[0] / rectified_point_homogeneous[2], 
                       rectified_point_homogeneous[1] / rectified_point_homogeneous[2])

    return rectified_point

def rectify_images(img_left, img_right, K1, D1, K2, D2, R, T, image_size):
    # Stereo rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T)

    # Compute the rectification transforms for each camera
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    # Apply the rectification transforms
    rectified_left = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)

    return rectified_left, rectified_right, Q


def find_rotation_matrix_and_draw_matches(img1, img2, K):
    # Step 1: Detect ORB features and compute descriptors.
    # orb = cv2.ORB_create()
    # keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    # keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # # Step 2: Match features using FLANN matcher.
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(descriptors1, descriptors2)
    # matches = sorted(matches, key=lambda x: x.distance)
    
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

   # Create a BFMatcher object with default params
    bf = cv2.BFMatcher()

    # Match descriptors using KNN algorithm with k=2
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good_matches.append(m)

    # Step 3: Extract matched keypoints.
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
    
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img_matches)
    plt.show()

    # Step 4: Estimate the fundamental matrix.
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    
    # Step 5: Compute the essential matrix.
    E = K.T @ F @ K
    
    # Step 6: Recover the pose (rotation and translation).
    _, R, T, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    
    return R, T,keypoints1, keypoints2, matches, mask

def detect_and_match_keypoints_sift(img1, img2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

   # Create a BFMatcher object with default params
    bf = cv2.BFMatcher()

    # Match descriptors using KNN algorithm with k=2
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            

    # Get matched keypoints
    matched_keypoints1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    matched_keypoints2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
    
    # Estimate perspective transform using matched keypoints
    H, mask = cv2.findHomography(matched_keypoints2, matched_keypoints1, cv2.RANSAC)
   
    
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.imshow(img_matches)
    plt.show()
    
    # Rectify images based on estimated homography
    rectified_img1 = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))
    rectified_img2 = cv2.warpPerspective(img2, H, (img2.shape[1], img2.shape[0]))

    return rectified_img1, rectified_img2, H               
                
                
 ###Read images               
img1 = cv2.imread('0000_Speckle1.bmp')
img2 = cv2.imread('0000_Speckle3.bmp')


 # Find contours in the binary image
circular_contours1,circular_contours_centers1 = detectMarkers(img1)
circular_contours2,circular_contours_centers2 = detectMarkers(img2)

# Draw contours on the original image
output_image = img1.copy()
cv2.drawContours(output_image, circular_contours1, -1, (0, 255, 0), 2)
output_image_resized = cv2.resize(output_image,(0, 0),fx=0.25,fy=0.25)
cv2.imshow("detected markers img1", output_image_resized)

output_image2 = img2.copy()
cv2.drawContours(output_image2, circular_contours2, -1, (0, 255, 0), 2)
output_image_resized2 = cv2.resize(output_image2,(0, 0),fx=0.25,fy=0.25)
cv2.imshow("detected markers img2", output_image_resized2)


## stereo matching and disparity calculation
# Example calibration parameters
### Rectify images by estimating camera parameters
baseline = 350.0
focal_length_mm = 16.24
pixel_size = 0.0055
# focal_length = focal_length_mm/pixel_size
# K1 = np.array([[focal_length, 0, img1.shape[1] / 2],
#                 [0, focal_length, img1.shape[0] / 2],
#                 [0, 0, 1]])  # Intrinsic matrix of left camera
# D1 = np.zeros(5)  # Distortion coefficients of left camera
# K2 = np.array([[focal_length, 0, img2.shape[1] / 2],
#                 [0, focal_length, img2.shape[0] / 2],
#                 [0, 0, 1]])  # Intrinsic matrix of right camera
# D2 = np.zeros(5)  # Distortion coefficients of right camera
# image_size = (img1.shape[1], img1.shape[0])
# R, T, keypoints1, keypoints2, matches, mask = find_rotation_matrix_and_draw_matches(img1, img2, K1)
# rectified_left, rectified_right, Q = rectify_images(img1, img2, K1, D1, K2, D2, R, T, image_size)

#### Rectification Using homography
rectified_left, rectified_right, H = detect_and_match_keypoints_sift(img1, img2)
rectified_left_resized = cv2.resize(rectified_left,(0, 0),fx=0.25,fy=0.25)
cv2.imshow('rectified_left_resized', rectified_left_resized)

rectified_right_resized = cv2.resize(rectified_right,(0, 0),fx=0.25,fy=0.25)
cv2.imshow('rectified_right_resized', rectified_right_resized)

#### Compute disparity based on rectified images
# stereo = cv2.StereoBM_create(numDisparities=16*9, blockSize=5)
blockSize = 5
min_disp = 3
num_disp = 16 * 11
mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                numDisparities=num_disp,
                                blockSize=blockSize,
                                disp12MaxDiff=25,
                                uniquenessRatio=1,
                                speckleWindowSize=2,
                                speckleRange=1,
                                mode=mode)

# Compute the disparity map
disparity = stereo.compute(rectified_left, rectified_right).astype(np.float32) / 16.0
disparity[disparity < 0] = 0.1
# Compute disparity maps
disparity_left = stereo.compute(rectified_left, rectified_right).astype(np.float32) / 16.0
disparity_right = stereo.compute(rectified_right, rectified_left).astype(np.float32) / 16.0

# # Avoid division by zero by setting zero disparities to a small value
disparity_left[disparity_left <= 0] = 0.1
disparity_right[disparity_right <= 0] = 0.1


# Normalize the disparity map for visualization
disparity_norm = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_norm = np.uint8(disparity_norm)
disparity_norm_resized = cv2.resize(disparity_norm,(0, 0),fx=0.25,fy=0.25)
cv2.imshow('Disparity Map', disparity_norm_resized)
# cv2.imwrite('disparity_norm.png',disparity_norm)

####Postprocess disparity map
# # Create WLS filter
# lmbda = 8.0
# sigma = 1.5
# median_filtered = cv2.medianBlur(disparity_norm, 5)
# median_filtered_float = np.float32(median_filtered)
# wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
# wls_filter.setLambda(lmbda)
# wls_filter.setSigmaColor(sigma)

# # # Apply WLS filter to the disparity map
# disparity_filtered = wls_filter.filter(disparity, grayImg1, None, median_filtered_float)

# # ###Normalize and display the filtered disparity map
# disparity_filtered_norm = cv2.normalize(disparity_filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
# disparity_filtered_norm_resized = cv2.resize(disparity_filtered_norm,(0, 0),fx=0.25,fy=0.25)
# cv2.imshow('Filtered Disparity Map', disparity_filtered_norm_resized)


# # Compute the depth map
depth_map = (baseline * focal_length_mm) / (pixel_size*disparity)
# depth_map_left = (focal_length_mm * baseline) / (pixel_size*disparity_left)
# depth_map_right = (focal_length_mm * baseline) / (pixel_size*disparity_right)


# # Display the depth map
# depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
# depth_map_normalized = np.uint8(depth_map_normalized)
# # Apply a colormap to the normalized depth map
# heatmap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
# heatmap_resized = cv2.resize(heatmap,(0, 0),fx=0.25,fy=0.25)
# cv2.imshow('Depth Map', heatmap_resized)


# ### Calculate depth of a particular marker
maker_position_left = circular_contours_centers1[5]
corresponding_point_in_rectified_img = find_corresponding_point(maker_position_left, H)
depth_val = depth_map[int(corresponding_point_in_rectified_img[0]),int(corresponding_point_in_rectified_img[1])]
print('depth_val:',depth_val)

cv2.waitKey(0)
cv2.destroyAllWindows()