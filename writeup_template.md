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

[image1]: ./output_images/undist_example.png "Undistorted"
[image2]: ./output_images/test_img_example.jpg "Road Transformed"
[image3]: ./output_images/threshold_example.jpg "Binary Example"
[image4]: ./output_images/warp_example.jpg "Warp Example"
[image5]: ./output_images/poly_fit_example.jpg "Fit Visual"
[image6]: ./output_images/result_example.jpg "Output"
[video1]: ./project_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

You're reading it!

### Camera Calibration

#### Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function camera_calibration of the file called `lnfdr.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

I used a combination of color and  sobel x gradient, gradient direction thresholds to generate a binary image (thresholding steps at lines 360 through 373 in function pipeline file `lnfdr.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)
I tried to use Sobel y, magnitude gradients and other color channels like H channel but the one i use is the best result i got.
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 154 through 174 in the file `lnfdr.py`.  The `warp()` function takes as inputs an image (`img`) and it creates the source and dist points.  I chose the hardcode the source and destination points in the following manner:

```python
vertices_src = np.array([[float(img.shape[1]*.10),img.shape[0]], 
					 [float(img.shape[1]*.90),img.shape[0]],
					 [float(img.shape[1]*.57),float(img.shape[0]*.63)],
					 [float(img.shape[1]*.43),float(img.shape[0]*.63)]],np.float32)
vertices_dst = np.array([[float(img.shape[1]*.20),img.shape[0]], 
						 [float(img.shape[1]*.80),img.shape[0]],
						 [float(img.shape[1]*.80),0],
						 [float(img.shape[1]*.20),0]],np.float32)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 128, 720      | 256, 720      | 
| 1152, 720     | 1024, 720     |
| 729, 453      | 1024, 0       |
| 550, 453      | 256, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial.

I used a histogram with the sliding window search for the very first image in the video to fit a second degree poly, the code for this part is in function blind_lane_search(). 
later on i used a margin search given the pervious poly fit to locate the new lanes and re fit the second degree poly, the code is in function follow_up_lane_search().
an example for the poly fitting is:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
​​ 
I did this in function calc_rad_and_offset().

For the radius, I used the equation described in this part t calculate the right and left curvature radius in the pixel space:

```python
left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

then i transformed it into meter space using the factors:
```python
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

and then i return thhe average between the left and right radii.

For the offset, i get the bttom most x position of the lanes using the poly like:
```python
left_lane_bttom = left_fit[0]*image_size[0]**2 + left_fit[1]*image_size[0] + left_fit[2]
right_lane_bottom = right_fit[0]*image_size[0]**2 + right_fit[1]*image_size[0] + right_fit[2]

then i consider the ccenter position of the lanes as the average position between the 2 positions left_lane_bttom and right_lane_bottom.
later, i subtract the center of the image in x space with the center of the lanes to calculate the offset:
```python
offset = (((left_lane_bttom + right_lane_bottom)/2.) - (image_size[1]/2.)) * xm_per_pix

same for the offset, i transform it from pixel to meter space.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function `display_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
I used the approach described in the lesson by doing camera calibration, undistortion, gradient and color thresholding, and perspective transform to find the lanes.
I used a combination between Sobel x grad, direction gradient anf S channel thresholds to accurately find the lanes.
What could fail, is the use cases where the road surface is a bit pumpy as well as when the vehicle is changing lanes.
To improve the accuracy of my pipeline, i can consideer the history poly fitting to average the current one to smot in the lanes, radius and offset.


