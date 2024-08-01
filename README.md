# StereoImaging

## Problem Statement:
Given two images forming a stereo pair, each capturing different reflective markers, the goal is to detect these reflective markers in the 2D images and determine their 3D positions in the real world. The reflective markers may vary in shape, size, and position. For this assignment, the objective is to calculate the 3D position of only the marked reflector.

## Input:
1. Two images of the same setup with different view angles.
2. Camera parameters: 
   - Baseline = 350mm
   - Focal length = 16.24mm
   - Pixel size (in X and Y) = 0.0055mm

## Assumptions:
Following are the assumptions during the design of the solution:
1. Cameras are calibrated and given images are corrected.
2. There are no lens distortions.
3. The relative positions of all the markers are known, this is to identify a particular marker for depth calculation.

## Solution
### Step 1: Detect Markers
Approaches used:
- Contour detection
- Maximum intensity (bright) pixel detection
- Hough circle identification

The code for these approaches is in the `DetectMarkers.py` file.

### Step 2: Rectify Images
Defining the epipolar geometry restricts the search for corresponding points to epipolar lines, reducing the complexity and increasing the accuracy of stereo matching. To simplify disparity calculation, the images should be on the same linear plane, meaning the epipolar lines should be horizontal. The rectification process transforms the images so that corresponding points in the left and right images lie on the same horizontal line, making the images appear as if they were captured with a parallel stereo setup.

Steps for rectification:
1. **Feature extraction**: SIFT algorithm is used for feature extraction.
2. **Match extracted features** in both images.
3. **Remove outliers** and consider the best features only.
4. **Rectify images** using homography or by calculating camera extrinsic parameters.

#### Using Homography:
Estimate perspective transform using matched keypoints and rectify images based on estimated homography. The code for this approach is in the function `detect_and_match_keypoints_sift` in the `DepthEstimation.py` file.

#### Estimate Camera Parameters:
Using given parameters and with the assumption that there are no distortions in the cameras, intrinsic parameters are calculated. These calculated intrinsic parameters are used to rectify the images by determining the fundamental matrix, essential matrix, and the camera pose (rotation and translation). The rectified images ensure that the corresponding points in the left and right images are aligned horizontally, facilitating accurate disparity computation and depth estimation. The code for this approach can be found in the functions `find_rotation_matrix_and_draw_matches` and `rectify_images` in the `DepthEstimation.py` file.

### Step 3: Compute Disparity
Using rectified images from Step 2, compute the disparity. Images rectified using homography are used for this calculation. To compute disparity, OpenCV’s `cv2.StereoSGBM` and `cv2.StereoBM` functions are tried.

#### StereoBM (Block Matching):
- **Block Matching**: Compares blocks of pixels between the left and right images.
- **Sum of Absolute Differences (SAD)**: Minimizes the sum of absolute differences between the pixel intensities in corresponding blocks.

#### StereoSGBM (Semi-Global Block Matching):
- **Semi-Global Matching (SGM)**: Considers both left-to-right and right-to-left consistency.
- **Cost Aggregation**: Aggregates matching costs over multiple directions to improve disparity estimation accuracy.

#### Choosing Between StereoBM and StereoSGBM:
- **Speed vs. Accuracy**: Use StereoBM for real-time applications with simpler scenes where speed is crucial. Use StereoSGBM for applications requiring higher accuracy and handling complex scenes at the cost of computation time.
- **Scene Complexity**: If the scene includes textureless regions, occlusions, or varying lighting conditions, StereoSGBM is likely to provide more reliable disparity maps.
- **Parameter Tuning**: Both algorithms have parameters (numDisparities, blockSize, etc.) that can be adjusted based on the specific characteristics of your stereo setup and application requirements.

In summary, StereoBM is faster but less accurate in challenging scenes, while StereoSGBM offers better accuracy at the expense of computational speed. In this assignment, StereoSGBM provided better results than StereoBM.

### Step 4: Calculate Depth
Using the disparity map and depth estimation formula, the depth map is calculated. To find the depth at a particular point in an input 2D image, the point’s corresponding location in the rectified image is calculated using a homography matrix. The value at this location in the depth map represents the depth, which indicates the distance of that point from the camera unit.
