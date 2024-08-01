# StereoImaging
### Problem statement:
	Given two images forming a stereo pair, each capturing different reflective markers, the goal is to detect these reflective markers in the 2D images and determine their 3D positions in the real world. The reflective markers may vary in shape, size, and position. For this assignment, the objective is to calculate the 3D position of only the marked reflector.
 
#### Input:
  1. Two images of same setup with different view angles.
  2. Camera parameters: Baseline = 350mm, Focal length = 16.24mm, Pixel size (in X and Y) = 0.0055mm
#### Assumptions:
Following are the assumptions dusring designing the solution
1.	Cameras are calibrated and given images are corrected
2.	There are no any lens distortions
3.	The relative positions of all the markers are known this is to identify a particular marker for depth calculation
#### Solution
##### Step1: Detect markers
  Approach 1: Contour detection
  Approach 2: Maximum intensity (bright) pixel detection
  Approach 3: Hough circle identification
  All these approaches are in DetectMarkers.py file

##### Step2: Rectify images
Defining the epipolar geometry restricts the search for corresponding points to epipolar lines, reducing the complexity and increasing the accuracy of stereo matching. To simplify disparity calculation, the images should be on the same linear plane, meaning the epipolar lines should be horizontal. The rectification process transforms the images so that corresponding points in the left and right images lie on the same horizontal line, making the images appear as if they were captured with a parallel stereo setup.
Following are the steps used for rectification:
a.	Feature extraction: SIFT algorithm is used for feature extraction.
b.	Match extracted features in both images
c.	Remove outliers and consider best features only
d.	Once keyfeatures are detected; the images can be rectified using homography or by calculating camera extrinsic parameters.
      i.	Using homography:
         Estimate perspective transform using matched keypoints and rectify images based on estimated homography. The code for this approach is in the function “detect_and_match_keypoints_sift” in "DepthEstimation.py"file.
      ii.	Estimate camera parameters:
        Using given parameters and with the assumption that there are no distortions in the cameras following intrinsic parameters are calculated.
baseline = 350.0
focal_length_mm = 16.24
pixel_size = 0.0055
focal_length = focal_length_mm/pixel_size
K1 = np.array([[focal_length, 0, img1.shape[1] / 2],
                [0, focal_length, img1.shape[0] / 2],
                [0, 0, 1]])  # Intrinsic matrix of left camera
D1 = np.zeros(5)  # Distortion coefficients of left camera
K2 = np.array([[focal_length, 0, img2.shape[1] / 2],
                [0, focal_length, img2.shape[0] / 2],
                [0, 0, 1]])  # Intrinsic matrix of right camera
D2 = np.zeros(5)  # Distortion coefficients of right camera

The fundamental matrix is calculated using matched keypoints.
By decomposing the fundamental matrix, the essential matrix is computed.
Using essential matrix and camera intrinsic parameters camera pose (rotation & translation) is calculated.
Using these estimated extrinsic values (rotation & translation) and intrinsic parameters (K1, K2) images are rectified.
The code for this approach is in the function “find_rotation_matrix_and_draw_matches” and “rectify_images” in "DepthEstimation.py"file.

##### Step3: Compute disparity
Using rectified images in step 2 compute the disparity. Images rectified using homography are used for this calculation.
To compute disparity opencv’s cv2.StereoSGBM and cv2.StereoBM functions are tried.
StereoBM (Block Matching)
○	Block Matching: StereoBM (Block Matching) computes disparities by comparing blocks of pixels between the left and right images.
○	Sum of Absolute Differences (SAD): It calculates the disparity by minimizing the sum of absolute differences between the pixel intensities in corresponding blocks.
StereoSGBM (Semi-Global Block Matching)
○	Semi-Global Matching (SGM): StereoSGBM uses a semi-global matching algorithm that considers both left-to-right and right-to-left consistency.
○	Cost Aggregation: It aggregates matching costs over multiple directions (disparities and scanlines) to improve the accuracy of disparity estimation.
Choosing Between StereoBM and StereoSGBM
●	Speed vs. Accuracy: Use StereoBM for real-time applications with simpler scenes where speed is crucial. Use StereoSGBM for applications requiring higher accuracy and handling of complex scenes at the cost of computation time.
●	Scene Complexity: If your scene includes textureless regions, occlusions, or varying lighting conditions, StereoSGBM is likely to provide more reliable disparity maps.
●	Parameter Tuning: Both algorithms have parameters (numDisparities, blockSize, etc.) that can be adjusted based on the specific characteristics of your stereo setup and application requirements.
In summary, StereoBM is faster but less accurate in challenging scenes, while StereoSGBM offers better accuracy at the expense of computational speed. 
In this assignment StereoSGBM gave better results than StereoBM.

##### Step4: Calculate depth
Using this disparity map and depth estimation formula, depth map is calculated.
To find the depth at a particular point in an input 2D image, the point’s corresponding location in the rectified image is calculated using a homography matrix. The value at this location in the depth map represents the depth, which indicates the distance of that point from the camera unit.






