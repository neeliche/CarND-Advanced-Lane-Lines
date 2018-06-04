**Advanced Lane Finding Project**

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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[Camera_Calibration_Chessboard]: ./output_images/Camera_Calibration_Chessboard.png "Camera_Calibration_Chessboard"
[Combined_Threshold_Yellow_line_difficulty]: ./output_images/Combined_Threshold_Yellow_line_difficulty.png "Combined_Threshold_Yellow_line_difficulty"
[Final_Lane_Data_for_frame]: ./output_images/Final_Lane_Data_for_frame.png "Final_Lane_Data_for_frame"
[HLS_Thresholding]: ./output_images/HLS_Thresholding.png "HLS_Thresholding"
[Lane_Data_imposed_on_image]: ./output_images/Lane_Data_imposed_on_image.png "Lane_Data_imposed_on_image"
[Lane_detection_color_overlap]: ./output_images/Lane_detection_color_overlap.png "Lane_detection_color_overlap"
[Lane_line_Polynomial]: ./output_images/Lane_line_Polynomial.png "Lane_line_Polynomial"
[Perspective_transform_birds_eye_view]: ./output_images/Perspective_transform_birds_eye_view.png "Perspective_transform_birds_eye_view"
[Sample_image_Undistorted]: ./output_images/Sample_image_Undistorted.png "Sample_image_Undistorted"
[Sample_Sobel_threshold_image]: ./output_images/Sample_Sobel_threshold_image.png "Sample_Sobel_threshold_image"
[Sliding_windows_lane_detection]: ./output_images/Sliding_windows_lane_detection.png "Sliding_windows_lane_detection"
[Threshold_Masked_Image]: ./output_images/Threshold_Masked_Image.png "Threshold_Masked_Image"
[Chessboard_corners_detection]: ./output_images/Chessboard_corners_detection.png "Chessboard_corners_detection"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./P4-Advanced-Lane-Detection.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

A sample of the corners that it detected for all the sample training images that were provided:

![Chessboard_corners_detection][Chessboard_corners_detection]

As you can see, not all the images had the corners detected in them. I am attributing it to the nature of the image and the capabilities of the OpenCV package in finding the edges accurately.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![Camera_Calibration_Chessboard][Camera_Calibration_Chessboard]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

After performing the distortion correction based on the previous set of camera calibration performed, the undistorted image looks like the following:

![Sample_image_Undistorted][Sample_image_Undistorted]

You can see the undistortion on the red car bonnet. In the original image its pretty much straight. In the distortion corrected image, its bent like how a bonnet would look like from the inside of a car.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

At first I wanted to look at the various types of thresholds that were mentioned in the lectures. Performing each of the Sobel threshold, gradient threshold, direction threshold and combining all together, helped me find the lane lines in the quiz image:

![Sample_Sobel_threshold_image][Sample_Sobel_threshold_image]

But, when I tried it with the above test image which contains a yellow line on the left side, it was pretty difficult to find the lane lines:

![Combined_Threshold_Yellow_line_difficulty][Combined_Threshold_Yellow_line_difficulty]

This is when I realized that the reason for using a different color space was because of how much of data grayscale conversion looses out. Thus converting the image into a HLS color space and performing Sobel threshold on it, helped me get the required lane lines as shown below:

![HLS_Thresholding][HLS_Thresholding]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The next logical step was to create a mask on the region of interest. This helped me visualize the presence of the lane line better. You can find these in the ipynb file under the method name
```python
def apply_region_of_interest(image, vertices):
```


Ex:
![Threshold_Masked_Image][Threshold_Masked_Image]

I then used the following `src` and `dst` coordinates to perform the perspective transformation. This section of the code can be found in the ipynb file under the section:
```python
def warp(img):
```
Where I used the following map values:

```python
src = np.float32(
    [[150, 720],
     [550, 480],
     [750, 480],
     [1200, 720]])

dst = np.float32(
    [[200,720],
     [200,0],
     [1080,0],
     [1080,720]])
```

Once the image was warped, it would look like the following:

![Perspective_transform_birds_eye_view][Perspective_transform_birds_eye_view]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Here I used the sliding windows approach thought to us in the lectures that is based on the following equation to find the lane lines.

* Divide the image into n 'windows'
* Take the histogram for this windows
* Based on the peaks, determine the points that would be the Lane

Refer to the code in the file under the method:
```python
def sliding_window_polynomial(binary_warped):
```

If we have the guestimate on where the lines are currently, we can reuse them to search around that area to find the lines in the new image frame. Refer to:

```python
def skip_windows_polynomial(binary_warped, left_fit, right_fit):
```

Performing this and visualizing the clusters of data points reveals a sample like this:

![Sliding_windows_lane_detection][Sliding_windows_lane_detection]

Using the following method, I was able to calculate the polynomial for each lane:

```python
def lane_poly(yval, poly_coeffs):
    """Returns x value for poly given a y-value.
    Note here x = Ay^2 + By + C."""
    return poly_coeffs[0]*yval**2 + poly_coeffs[1]*yval + poly_coeffs[2]
```

This results in a sample like:

![Lane_line_Polynomial][Lane_line_Polynomial]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Here I took some "inspiration" from the lecture codes and ended up with the code to find the curvature with the following equations once I got the lane polynomial for the given lane fit:

```python
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
center = (((left_fit[0]*720**2+left_fit[1]*720+left_fit[2]) +(right_fit[0]*720**2+right_fit[1]*720+right_fit[2]) ) /2 - 640)*xm_per_pix
```

You can find the entire code in the method

```python
def get_curvature_center(binary_warped, left_fit, right_fit):
```



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here, I used the `Minv` calculated when I performed perspective transformation on the color image block where I 'guestimate' the line to be present.

Code for it:

```python
warp_zero = np.zeros_like(warp_img).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
# Combine the result with the original image
result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
plt.imshow(result)
```

Once I had all the information, the information was added to the frame like:

![Final_Lane_Data_for_frame][Final_Lane_Data_for_frame]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_output.mp4) (project_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The information from the lectures was quite sufficient for me to get started with. But, I soon ran into several problems in trying to understand why my lanes were not being detected correctly and how to store the best fit and making use of the suggested ``class Line``. After some googling around and stumbling across some gem of a Git repositories, I was able to get some more traction on it and I was able to complete the video.

Shortfalls:
* Does not work on changing lighting conditions:
  - I would like to perform a varying threshold for the Sobel threshold and HLS threshold till I get the lines for each frame if the default one does not detect it. This is useful in conditions where the image's lighting condition varies quite a bit.
* Does not work on overly curved roads:
  - This I attribute to the fixes area that I mask to get the lane lines. This needs to be dynamic once again to get the lanes for overly curvy roads. This is something that I would like to spend more time in the future to get it done better.
* Maintaining the history of the lane information:
  - This is something that I have done in quite a primitive manner. Some of the checks are good, but I am not currently checking on situations where I have a good Left lane but the right lane detected has a very small radius and so on. This might help me solve the harder video much better.


There could be many other shortcomings that are present with this current implementation. This, I will be able to better judge once I have more knowledge on processing these images better.
