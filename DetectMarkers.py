# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:08:51 2024

@author: prana
"""
import cv2
import numpy as np

def findContours(closed_bin_img1):
    contours, _ = cv2.findContours(closed_bin_img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

     # Filter contours to keep only circular ones
    circular_contours = []
    for contour in contours:
        # Calculate contour area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate circularity
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        # print(circularity)
        # Circularity threshold
        if 0.85 < circularity < 1.2 and area > 20:
                circular_contours.append(contour)
                print('circularity:',circularity)
                print('area:',area)
                
    circular_contours_centers =[]
    for i in circular_contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            circular_contours_centers.append([cx,cy])
            
    return circular_contours, circular_contours_centers

def findBrightestSpots(closed_bin_img,img):
    brightest_spots = []
    num_spots=150
    brightest_spots_img = img.copy()
    for _ in range(num_spots):
        # Find the minimum and maximum intensity values and their locations
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(closed_bin_img)
        
        # Draw a circle around the brightest spot
        cv2.circle(brightest_spots_img, max_loc, 10, (0, 0, 255), 2)
        
        # Save the spot location and value
        brightest_spots.append((max_loc, max_val))
        
        # Zero out the maximum value location to find the next brightest spot
        cv2.circle(closed_bin_img, max_loc, 10, (0, 0, 0), -1)
    return brightest_spots, brightest_spots_img

def detectCircleHough(closed_bin_img):
    dp=1.2
    min_dist=20
    param1=100
    param2=10
    min_radius=2
    max_radius=15
    circles1 = cv2.HoughCircles(closed_bin_img, cv2.HOUGH_GRADIENT, dp, min_dist,
                                param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    if circles1 is not None:
        circles1 = np.uint16(np.around(circles1))
        circles1 = circles1[0, :]
        # Filter circles to keep unique ones
        unique_circles = []
        for circle in circles1:
            x, y, r = circle
            is_unique = True
            for uc in unique_circles:
                ux, uy, ur = uc
                distance = np.sqrt((x - ux)**2 + (y - uy)**2)
                if distance < min_dist:
                    is_unique = False
                    break
            if is_unique:
                unique_circles.append(circle)
    return unique_circles 


# if __name__ == '__main__':
def detectMarkers(img):
    
    ###convert to gray
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve detection
    ksize = 3
    blurred = cv2.GaussianBlur(grayImg, (ksize, ksize), 0)
    # blurred_resized = cv2.resize(blurred,(0, 0),fx=0.25,fy=0.25)
    # cv2.imshow('blurred_resized',blurred1_resized)    
    
    ###Thresholding
    thresh = 254 #set high value as markers are bright spots
    # ret, bin_img = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)
    ret, bin_img = cv2.threshold(grayImg, thresh, 255, cv2.THRESH_BINARY)
    
    # bin_img_resized = cv2.resize(bin_img,(0, 0),fx=0.25,fy=0.25)
    # cv2.imshow('bin_img_resized',bin_img1_resized)
    
    ##Morphological operations
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel,iterations=1)
    # closed_bin_img_resized = cv2.resize(closed_bin_img,(0, 0),fx=0.25,fy=0.25)
    # cv2.imshow('closed_bin_img_resized',closed_bin_img1_resized)
    
     # Find contours in the binary image
    circular_contours,circular_contours_centers = findContours(closed_bin_img)
    
    ####Detect markers using hogh transform for circle detection
    # unique_circles = detectCircleHough(closed_bin_img)
    # # If some circles are detected, draw them on the original image
    # hough_output_image = img.copy()            
    # for i in unique_circles:
    #     # Draw the outer circle
    #     cv2.circle(hough_output_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #     # Draw the center of the circle
    #     cv2.circle(hough_output_image, (i[0], i[1]), 2, (0, 0, 255), 3)

    # #Display the results
    # hough_output_image_resized = cv2.resize(hough_output_image,(0, 0),fx=0.25,fy=0.25)
    # cv2.imshow("Detected Circles Hough Image", hough_output_image_resized) 
  

    # ###Find the brightest spots
    # brightest_spots, brightest_spots_img = findBrightestSpots(grayImg,img)
    # # # Display the results
    # brightest_spots_img_resized = cv2.resize(brightest_spots_img,(0, 0),fx=0.25,fy=0.25)
    # cv2.imshow("Brightest Spots", brightest_spots_img_resized)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return circular_contours, circular_contours_centers