
**Advanced Lane Finding Project**

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted_chess_image.png "Undistorted chessboard image"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/color_grad_img.png "Binary Example"
[image4]: ./output_images/persp_xform.png "Warp Example"
[image5]: ./output_images/test1._lane_fit.png "Fit Visual"
[image6]: ./output_images/lane_id_curved_stats.png "Output"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `get_calibration_coeffs` method of `AdvancedLaneFinding` class in cell 2 of the IPython notebook located in "./AdvancedLaneFinding.ipynb"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Distortion-corrected image.

The distortion correction code can be found in the method `undistort_img` method in `AdvancedLaneFinding` class. To demonstrate this step, I applied the `cv2.undistort()` to one of the test images like this one:
![alt text][image2]

#### 2. Color and gradient thresholded binary image.

First, Gradient threshold was calculated by applying sobel, magnitude and direction gradient to the input image after it was converted to gray scale. The code for these three calculations can be found in cells 8, 9 and 10 of the IPython notebook. A minimum threshold of 0 and max of 255 was used for all three computations with a kernel size of 3. The `combined_grad()` method applies all three gradients to the input image and returns a filtered image from the output of the three transforms. 

Color transforms are applied after gradient using both HLS and RGB channels. The S channel from HLS and R from RGB were selected since they do a better job at detecting lane lines of any color and width. Code for these two transforms is in cells 13 and 14 of the IPython notebook

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in methods `hls_thresh` and `rgb_thresh ` of the `AdvancedLaneFinding` class)

![alt text][image3]

#### 3. Perspective transform

Perspective transform for images is done by the method `transform`. The transform is done from the two methods on the input image by calling `cv2.warpPerspective()` method from the OpenCV library. 



Here is a [link to the cell](./AdvancedLaneFinding.ipynb#Camera-Calibration) containing source & destination points for warp

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Lane line identification and fitting to a polynomial

Fitting of the lanes to a polynomial and identifying the lanes was accomplished by first getting a histogram of the lane plots. From the histogram, peaks were extracted as the regions containg the pixel density for the lanes. Using the peaks, the baseline was calculated. A sliding window mechanism was used to detect the lanes as they curve along. The number of windows was set to 9. A loop was set up to iterate through each window and the coordinates were calculated for x-left, x-right, y-low and y-high points. Using these values, the left and right fit for the two lanes were set using the non zero values from the binary warped image matrix. This code is part of the `sliding_window_lane_fit_polynomial` method.

Here is the image of identified lane lines plotted with sliding windows:

![alt text][image5]

The rest of the images can be found in the output_images folder of this repo

#### 5. Radius of curvature classification

The radius of curvature calculation is contained in `calc_radius_curvature` method. The curvature is calculated using the values returned by `sliding_window_lane_fit_polynomial` method

#### 6. Plotting the image back to the lane lines.

The plotted lanes are projected back to the road using the code in the `lane_area_identify` method. This method takes the original image, the binary image from the pipeline, the left and right fit from `sliding_window_lane_fit_polynomial` and the inverse perspective transform. The image is plotted using `fillPoly` and `polyLines` methods from the OpenCV library

![alt text][image6]

---

### Pipeline (video)


Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion


Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

1.  The calibration and distortion correction measurements were straightforward but a lot of time had to be spend figuring out source and destination points for the perspective transform. I had to try out several values for the destination points since most of the initial values only worked on a subset of the images. The code in `sliding_window_lane_fit_polynomial` and `line_fit_from_prev_frame` came really handy in testing various values out since they gave an accurate projection of whether the selected points were a good fit for all the images

2. The lane area identification was another challenge since calculating an accurate fit for all the different angles and measurements was tedious and took several iterations before I could finalize the equations

3. Applying the model on the challenge video shows the projected lanes having a wide variation from the actual road for sharp turns. So this model will have some issues on twisting & turning lanes. It can be resolved by more finely tweaking the destination points for the warp such that the lane projection is accurate
