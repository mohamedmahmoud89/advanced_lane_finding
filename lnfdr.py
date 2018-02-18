import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from moviepy.editor import VideoFileClip

# performs the camera calibration, image distortion correction and 
# returns the undistorted image
def camera_calibration(images):
	objpoints = []
	imgpoints = []
	objp = np.zeros((6*9,3),np.float)
	objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
	
	for fname in images:
		img = cv2.imread(fname)
		
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		shape = gray.shape[::-1]
		ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
		
		if ret == True:
			objp = objp.astype('float32')
			corners = corners.astype('float32')
			imgpoints.append(corners)
			objpoints.append(objp)
			img = cv2.drawChessboardCorners(img, (9,6), corners, ret)

	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

	return mtx,dist

# performs Sobel gradient thresholding
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
	# Apply the following steps to img
	# 1) Convert to grayscale
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# 2) Take the derivative in x or y given orient = 'x' or 'y'
	if orient == 'x':
		sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
	else:
		sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
	# 3) Take the absolute value of the derivative or gradient
	abs_sobel = np.absolute(sobel)
	# 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
	abs_sobel = np.uint8((abs_sobel*255)/np.max(abs_sobel))
	# 5) Create a mask of 1's where the scaled gradient magnitude 
			# is > thresh_min and < thresh_max
	grad_binary = np.zeros_like(abs_sobel)
	grad_binary[(abs_sobel > thresh[0]) & (abs_sobel < thresh[1])] = 1
	# 6) Return this mask as your binary_output image
	# Apply threshold
	return grad_binary

# perform gradient magnitude thresholding
def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
	# Convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Take both Sobel x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Calculate the gradient magnitude
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	# Rescale to 8 bit
	scale_factor = np.max(gradmag)/255 
	gradmag = (gradmag/scale_factor).astype(np.uint8) 
	# Create a binary image of ones where threshold is met, zeros otherwise
	mag_binary = np.zeros_like(gradmag)
	mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
	# Apply threshold
	return mag_binary

# perform gradient direction thresholding
def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
	# Grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Calculate the x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Take the absolute value of the gradient direction, 
	# apply a threshold, and create a binary image result
	absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
	dir_binary =  np.zeros_like(absgraddir)
	dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

	# Apply threshold
	return dir_binary
	
# perform color thresholding
def hls_select(img, thresh=(0, 255),type='s'):
	hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	if type == 's':
		channel = hls[:,:,2]
	elif type == 'h':
		channel = hls[:,:,0]
	else:
		channel = hls[:,:,1]
	binary_output = np.zeros_like(channel)
	binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
	return binary_output

# to display images	
def show_image(img,bin_img,desc=''):
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
	f.tight_layout()
	ax1.imshow(img)
	ax1.set_title('Original Image', fontsize=50)
	ax2.imshow(bin_img, cmap='gray')
	ax2.set_title(desc, fontsize=50)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()
	
# to display region of interest, useful for obtaining the points for warping	
def region_of_interest(img, vertices):
	"""
	Applies an image mask.
	
	Only keeps the region of the image defined by the polygon
	formed from `vertices`. The rest of the image is set to black.
	"""
	#defining a blank mask to start with
	mask = np.zeros_like(img)   
	
	#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
		
	#filling pixels inside the polygon defined by "vertices" with the fill color	
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	
	#returning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

# to display region of interest, useful for obtaining the points for warping		
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
	"""
	`img` is the output of the hough_lines(), An image with lines drawn on it.
	Should be a blank image (all black) with lines drawn on it.
	
	`initial_img` should be the image before any processing.
	
	The result image is computed as follows:
	
	initial_img * α + img * β + λ
	NOTE: initial_img and img must be the same shape!
	"""
	return cv2.addWeighted(initial_img, α, img, β, λ)

# perform perspective transform		
def warp(img):
	vertices_dst = np.array([[float(img.shape[1]*.20),img.shape[0]], 
						 [float(img.shape[1]*.80),img.shape[0]],
						 [float(img.shape[1]*.80),0],
						 [float(img.shape[1]*.20),0]],np.float32)
#	vertices_src = np.array([[float(img.shape[1]*.14),img.shape[0]], 
#						 [float(img.shape[1]*.87),img.shape[0]],
#						 [float(img.shape[1]*.54),float(img.shape[0]*.63)],
#						 [float(img.shape[1]*.46),float(img.shape[0]*.63)]],np.float32)

	vertices_src = np.array([[float(img.shape[1]*.10),img.shape[0]], 
					 [float(img.shape[1]*.90),img.shape[0]],
					 [float(img.shape[1]*.57),float(img.shape[0]*.63)],
					 [float(img.shape[1]*.43),float(img.shape[0]*.63)]],np.float32)
	# d) use cv2.getPerspectiveTransform() to get M, the transform matrix
	M = cv2.getPerspectiveTransform(vertices_src,vertices_dst)
	Minv = cv2.getPerspectiveTransform(vertices_dst, vertices_src)
	# e) use cv2.warpPerspective() to warp your image to a top-down view
	warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)

	return warped,M,Minv
	
#perform slide window search and fits a poly
def blind_lane_search(binary_warped):
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Draw the windows on the visualization image
		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	return ploty,left_fitx,right_fitx,left_fit,right_fit

# perform margin search based on a given poly and fits a poly	
def follow_up_lane_search(binary_warped,left_fit,right_fit):
	# Assume you now have a new warped binary image 
	# from the next frame of video (also called "binary_warped")
	# It's now much easier to find line pixels!
	nonzero = binary_warped.nonzero()
	
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 100
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
	
	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	
	return ploty,left_fitx,right_fitx,left_fit,right_fit

# display the output images with curv, offset and lanes on top	
def display_lane(undist,warped,ploty,left_fitx,right_fitx,Minv,radius,offset):
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0])) 
	# Combine the result with the original image
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
	
	if offset < 0:
		side = " right of center"
	else:
		side = " left of center"
	
	cv2.putText(result,"Radius = " + str(round(radius, 2)) + "(m)", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 10,thickness=5)
	cv2.putText(result,"Offset = " + str(round(abs(offset), 2)) + "m" + side, (100,200), cv2.FONT_HERSHEY_SIMPLEX, 2, 10,thickness=5)

	return result

# calculates the curvature radius and the offset	
def calc_rad_and_offset(ploty,left_fit,right_fit,left_fitx,right_fitx,image_size):
	# Define y-value where we want radius of curvature
	# I'll choose the maximum y-value, corresponding to the bottom of the image
	y_eval = np.max(ploty)
	left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
	right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension

	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	# Now our radius of curvature is in meters
	#print(left_curverad, 'm', right_curverad, 'm')
	
	radius = (left_curverad + right_curverad)/2.
	
	left_lane_bttom = left_fit[0]*image_size[0]**2 + left_fit[1]*image_size[0] + left_fit[2]
	right_lane_bottom = right_fit[0]*image_size[0]**2 + right_fit[1]*image_size[0] + right_fit[2]
	
	offset = (((left_lane_bttom + right_lane_bottom)/2.) - (image_size[1]/2.)) * xm_per_pix
	#print(offset, 'offset')
	
	return radius,offset
	

# Read the calibration images
images = glob.glob('camera_cal/calibration*.jpg')
# perform camera calibration only once
mtx,dist = camera_calibration(images)
# global vars
left_fit = []
right_fit = []
is_first_frame = True
# the complete pipeline
def pipeline(img):
	undist = cv2.undistort(img, mtx, dist, None, mtx)

	# Choose a Sobel kernel size
	ksize = 9 # Choose a larger odd number to smooth gradient measurements

	# Apply each of the thresholding functions
	gradx = abs_sobel_thresh(undist, orient='x', sobel_kernel=ksize, thresh=(20, 100))
	#grady = abs_sobel_thresh(undist, orient='y', sobel_kernel=ksize, thresh=(20, 80))
	#mag_binary = mag_thresh(undist, sobel_kernel=ksize, mag_thresh=(20, 80))
	dir_binary = dir_threshold(undist, sobel_kernel=3, thresh=(.7, 1.1))
	s_binary = hls_select(undist, thresh=(90, 255),type='s')
	#h_binary = hls_select(undist, thresh=(90, 200),type='h')
	#l_binary = hls_select(undist, thresh=(90, 200),type='l')

	#combined_1 = np.zeros_like(dir_binary)
	#combined_1[((gradx == 1) & (dir_binary == 1))] = 1

	combined_2 = np.zeros_like(s_binary)
	combined_2[(dir_binary == 1) & ((s_binary == 1) | (gradx == 1))] = 1

	#show_image(undist,gradx,'gradx')
	#show_image(undist,grady)
	#show_image(undist,mag_binary,'mag')
	#show_image(undist,dir_binary,'dir')
	#show_image(undist,s_binary,'s')
	#show_image(undist,combined_2,'combined')

	vertices = np.array([[int(undist.shape[1]*.10),undist.shape[0]], 
							 [int(undist.shape[1]*.90),undist.shape[0]],
							 [int(undist.shape[1]*.57),int(undist.shape[0]*.63)],
							 [int(undist.shape[1]*.43),int(undist.shape[0]*.63)]],np.int32)
	undist_crop = region_of_interest(undist,[vertices])
	undist_crop = weighted_img(undist_crop,undist)

	#show_image(undist,undist_crop,'cropping')

	waarped,M,Minv = warp(combined_2)

	global is_first_frame
	global right_fit
	global left_fit

	if is_first_frame == True:
		ploty,left_fitx,right_fitx,left_fit,right_fit = blind_lane_search(waarped)
		is_first_frame = False
	else:
		ploty,left_fitx,right_fitx,left_fit,right_fit = follow_up_lane_search(waarped,left_fit,right_fit)
	
	radius,offset = calc_rad_and_offset(ploty,left_fit,right_fit,left_fitx,right_fitx,waarped.shape)

	result = display_lane(undist,waarped,ploty,left_fitx,right_fitx,Minv,radius,offset)
	
	return result
###############################################################
# ues the pipeline to create the output video
white_output = 'project_video_out.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)