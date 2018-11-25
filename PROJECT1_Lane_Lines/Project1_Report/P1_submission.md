
# Self-Driving Car Engineer Nanodegree


## Project: **Finding Lane Lines on the Road** 
***
In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 

Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.

In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.

---
Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.

**Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**

---

**The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**

---

## 1. Project Detail

My project consists of following stages,

 1) Reading the image 
 2) Applied region selection
 3) Applied color selection
 4) Applied gray scaling. 
 5) Applied gaussian blur 
 6) Applied canny edge detection 
 7) Applied hough transform
 8) Applied lines and combined it with original image
 
 In order to achieve tihs, initialization of project is must and it was already provided and I did not changed anything in it.
 

<figure>
 <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
 </figcaption>
</figure>
 <p></p> 
<figure>
 <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
 </figcaption>
</figure>

**Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  

## Import Packages


```python
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
%matplotlib inline
```

## Step1: Read in an Image


```python
#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
```

    This image is: <class 'numpy.ndarray'> with dimensions: (540, 960, 3)





    <matplotlib.image.AxesImage at 0x7f9f0003f128>




![png](output_6_2.png)


## Ideas for Lane Detection Pipeline

**Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**

`cv2.inRange()` for color selection  
`cv2.fillPoly()` for regions selection  
`cv2.line()` to draw lines on an image given endpoints  
`cv2.addWeighted()` to coadd / overlay two images
`cv2.cvtColor()` to grayscale or change color
`cv2.imwrite()` to output images to file  
`cv2.bitwise_and()` to apply a mask to an image

**Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

## Helper Functions

Below are some helper functions to help get you started. They should look familiar from the lesson!


```python
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
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

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

#This function change all the colors other than white or yellow to black.
def color_selection_filter(ip_image):
    ysize = ip_image.shape[0];
    xsize = ip_image.shape[1];
    color_select_img = np.copy(ip_image);
    r_channel = ip_image[:,:,0];
    g_channel = ip_image[:,:,1];
    b_channel = ip_image[:,:,2];
    m_red=np.amax(r_channel);
    m_green=np.amax(g_channel);
    m_blue=np.amax(b_channel);
    min_blue= np.amin(b_channel);
    common_high_thresh = 0.70;
    blue_low_thresh = 1.25;
    red_threshold_high = common_high_thresh*m_red; 
    green_threshold_high = common_high_thresh*m_green;
    blue_threshold_high = common_high_thresh*m_blue;
    blue_threshold_low = blue_low_thresh*min_blue;
    blue_threshold_low = 100;
    selection =  (((r_channel > red_threshold_high) & \
                  (g_channel > green_threshold_high)));
    color_select_img[~selection] = [0,0,0]
    
    return color_select_img;
```


```python
#Let's consume provided functions.
os.listdir("test_images/")    
def main_function(img, show_hide_flag):
    #STEP-1: Print original image
    if(show_hide_flag):
        plt.rcParams['figure.figsize']=(16,20);
        sub = plt.subplot(1,2,1);
        sub.set_title("1. Provided image");
        plt.imshow(img);
    
    #STEP-2: Region selection
    imshape=img.shape;
    vertices = np.array([[(0,imshape[0]),(460,320),(510, 320), (imshape[1],imshape[0])]], dtype=np.int32);
    region_selection = region_of_interest(img,vertices);
    if(show_hide_flag):
        sub = plt.subplot(1,2,2);
        sub.set_title("2. Region selection applied");
        plt.imshow(region_selection);
        plt.show();
    
    #STEP 3: Color selection
    color_selection = color_selection_filter(region_selection);
    if(show_hide_flag):
        sub = plt.subplot(1,2,1);
        sub.set_title("3. Color selection applied");
        plt.imshow(color_selection);
    
    #STEP-4: convert to gray
    grayscale_img=grayscale(color_selection);
    if(show_hide_flag):
        sub = plt.subplot(1,2,2);
        sub.set_title("4. Gray scaling applied");
        plt.imshow(grayscale_img,cmap='gray');
        plt.show();
    
     #STEP 5: GaussianBlur
    kernel_size = 5
    blur_gray = gaussian_blur(grayscale_img,kernel_size);
    if(show_hide_flag):
        sub = plt.subplot(1,2,1);
        sub.set_title("5. GaussianBlur applied");
        plt.imshow(blur_gray,cmap='gray');
        
    
    #STEP 6: Canny edge detection
    low_threshold = 30
    high_threshold = 75
    canny_img=canny(blur_gray,low_threshold,high_threshold);
    if(show_hide_flag):
        sub = plt.subplot(1,2,2);
        sub.set_title("6. Canny edge detection applied");
        plt.imshow(canny_img,cmap='gray');
        plt.show();
    
    
    #STEP 7: Hough Transform
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 #minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments
    line_image = np.copy(img)*0 # creating a blank to draw lines on
    
    line_img = hough_lines(canny_img,rho, theta, threshold,min_line_length, max_line_gap);
    if(show_hide_flag):
        sub = plt.subplot(1,2,1);
        sub.set_title("7. Hough Transform applied");
        plt.imshow(line_img,cmap='gray');
        
    
    line_on_img = weighted_img(line_img,img);
    if(show_hide_flag):
        sub = plt.subplot(1,2,2);
        sub.set_title("8. Combined lines to original image");
        plt.imshow(line_on_img);
        plt.show();
    if(not(show_hide_flag)):        
        return line_on_img;

#img = plt.imread(IN_IMAGE_FOLDER+"/whiteCarLaneSwitch.jpg")
#img = plt.imread(os.path.join("ip_frames_solidWhiteRight","209.jpg"));
img = plt.imread('test_images/solidWhiteRight.jpg');
main_function(img, True)
```


![png](output_12_0.png)



![png](output_12_1.png)



![png](output_12_2.png)



![png](output_12_3.png)


## Test Images

Build your pipeline to work on the images in the directory "test_images"  
**You should make sure your pipeline works well on these images before you try the videos.**


```python
import os
import matplotlib.pyplot as plt
os.listdir("test_images/")
#Let's use above function to save images in 'test_images_output' folder.

if not os.path.exists('test_images_output'):
    os.makedirs('test_images_output');
img_path = [];
for file in os.listdir("test_images"):
    if file.endswith(".jpg") or file.endswith(".png"):
        img_path.append("test_images"+"/"+file);
for path in img_path:
    file = path.split("/")[len(path.split("/"))-1];
    img = plt.imread(path);
    output=main_function(img, False);
    plt.imsave('test_images_output'+"/"+file,output)
```

## Build a Lane Finding Pipeline



Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.

Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

Till now whatever I have created, with that I have achieved finding lane but marker on that are not solid one and what needed is not achieved. In order to get solid lines on both the side, lets modify solid_lines() function.


```python
# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
#Improving the draw_lines() functions.
#Modifying methods to draw straight line. 
def modified_solid_lines(img, lines, color=[255, 0, 0], thickness=2):
    THRESHOLD=20;
    xsize=img.shape[1];
    ysize=img.shape[0];
    left_lines=[];
    right_lines=[];
    for line in lines:
        for x1,y1,x2,y2 in line:
            m=(y2-y1)/(x2-x1);
            if(abs(m)>np.tan(THRESHOLD*np.pi/180)):
                if(x1<(xsize/2) and x2<(xsize/2)):
                    left_lines.append(line);
                else:
                    right_lines.append(line);
    
    final_lines=[];
    cordsx=[];
    cordsy=[];
    for line in left_lines:
        for x1,y1,x2,y2 in line:
            cordsx+=[x1,x2];
            cordsy+=[y1,y2];

    left_line=np.polyfit(cordsx,cordsy, 1);
    cordsx=[];
    cordsy=[];
    for line in right_lines:
        for x1,y1,x2,y2 in line:
            cordsx+=[x1,x2];
            cordsy+=[y1,y2];

    right_line=np.polyfit(cordsx,cordsy, 1);
    
    cv2.line(img,(int((ysize-left_line[1])//left_line[0]),ysize),(int((330-left_line[1])//left_line[0]),330),(255,0,0),10);
    cv2.line(img,(int((ysize-right_line[1])//right_line[0]),ysize),(int((330-right_line[1])//right_line[0]),330),(255,0,0),10);
    
def modified_hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    modified_solid_lines(line_img, lines)
    return line_img

os.listdir("test_images/")    
def modified_main_function(img):
    #STEP-1: Print original image
    #No need of it now.
    if(img.shape[2]==4): #Image with Alpha channel
        img=img[:,:,:3];
    
    
    #STEP-2: Region selection
    imshape=img.shape;
    vertices = np.array([[(0,imshape[0]),(450,330),(520, 330), (imshape[1],imshape[0])]], dtype=np.int32);
    region_selection = region_of_interest(img,vertices);
    
    #STEP 3: Color selection
    color_selection = color_selection_filter(region_selection);
    
    #STEP-4: convert to gray
    grayscale_img=grayscale(color_selection);
    
     #STEP 5: GaussianBlur
    kernel_size = 3
    blur_gray = gaussian_blur(grayscale_img,kernel_size);
    
    #STEP 6: Canny edge detection
    low_threshold = 30
    high_threshold = 75
    canny_img=canny(blur_gray,low_threshold,high_threshold);
       
    #STEP 7: Hough Transform
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 #minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments
    line_image = np.copy(img)*0 # creating a blank to draw lines on
    
    line_img = modified_hough_lines(canny_img,rho, theta, threshold,min_line_length, max_line_gap);
    
    line_on_img = weighted_img(line_img,img);
    return line_on_img;

img = plt.imread("test_images/whiteCarLaneSwitch.jpg")
plt.imshow(modified_main_function(img))
```




    <matplotlib.image.AxesImage at 0x7f9f0016e470>




![png](output_17_1.png)


## Test on Videos

You know what's cooler than drawing lanes over images? Drawing lanes over video!

We can test our solution on two provided videos:

`solidWhiteRight.mp4`

`solidYellowLeft.mp4`

**Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**

**If you get an error that looks like this:**
```
NeedDownloadError: Need ffmpeg exe. 
You can download it by calling: 
imageio.plugins.ffmpeg.download()
```
**Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**


```python
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
```


```python
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result=modified_main_function(image);
    return result
```

Let's try the one with the solid white lane on the right first ...


```python
from moviepy.editor import VideoFileClip
white_output = 'test_videos_output/solidWhiteRight.mp4'
    
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
try:
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    %time white_clip.write_videofile(white_output, audio=False)
finally:
    clip1.reader.close();
```

    [MoviePy] >>>> Building video test_videos_output/solidWhiteRight.mp4
    [MoviePy] Writing video test_videos_output/solidWhiteRight.mp4


    
      0%|          | 0/222 [00:00<?, ?it/s][A
      1%|          | 2/222 [00:00<00:12, 17.54it/s][A
      2%|▏         | 5/222 [00:00<00:11, 18.42it/s][A
      4%|▎         | 8/222 [00:00<00:11, 19.23it/s][A
      5%|▍         | 10/222 [00:00<00:11, 19.08it/s][A
      5%|▌         | 12/222 [00:00<00:11, 18.83it/s][A
      6%|▋         | 14/222 [00:00<00:10, 18.95it/s][A
      7%|▋         | 16/222 [00:00<00:10, 19.06it/s][A
      9%|▊         | 19/222 [00:00<00:10, 19.60it/s][A
      9%|▉         | 21/222 [00:01<00:10, 19.63it/s][A
     11%|█         | 24/222 [00:01<00:09, 19.92it/s][A
     12%|█▏        | 26/222 [00:01<00:09, 19.79it/s][A
     13%|█▎        | 29/222 [00:01<00:09, 19.94it/s][A
     14%|█▍        | 31/222 [00:01<00:09, 19.57it/s][A
     15%|█▌        | 34/222 [00:01<00:09, 20.20it/s][A
     17%|█▋        | 37/222 [00:01<00:09, 20.22it/s][A
     18%|█▊        | 40/222 [00:01<00:08, 20.40it/s][A
     19%|█▉        | 43/222 [00:02<00:11, 15.15it/s][A
     20%|██        | 45/222 [00:02<00:13, 13.20it/s][A
     21%|██        | 47/222 [00:02<00:14, 11.92it/s][A
     22%|██▏       | 49/222 [00:02<00:14, 11.66it/s][A
     23%|██▎       | 51/222 [00:03<00:14, 11.45it/s][A
     24%|██▍       | 53/222 [00:03<00:15, 11.15it/s][A
     25%|██▍       | 55/222 [00:03<00:15, 11.01it/s][A
     26%|██▌       | 57/222 [00:03<00:15, 10.97it/s][A
     27%|██▋       | 59/222 [00:03<00:14, 11.00it/s][A
     27%|██▋       | 61/222 [00:04<00:14, 11.01it/s][A
     28%|██▊       | 63/222 [00:04<00:14, 10.82it/s][A
     29%|██▉       | 65/222 [00:04<00:14, 10.97it/s][A
     30%|███       | 67/222 [00:04<00:14, 11.00it/s][A
     31%|███       | 69/222 [00:04<00:13, 11.07it/s][A
     32%|███▏      | 71/222 [00:04<00:13, 11.17it/s][A
     33%|███▎      | 73/222 [00:05<00:13, 11.19it/s][A
     34%|███▍      | 75/222 [00:05<00:13, 10.96it/s][A
     35%|███▍      | 77/222 [00:05<00:13, 11.13it/s][A
     36%|███▌      | 79/222 [00:05<00:12, 11.11it/s][A
     36%|███▋      | 81/222 [00:05<00:12, 11.10it/s][A
     37%|███▋      | 83/222 [00:05<00:12, 11.08it/s][A
     38%|███▊      | 85/222 [00:06<00:12, 11.09it/s][A
     39%|███▉      | 87/222 [00:06<00:12, 11.09it/s][A
     40%|████      | 89/222 [00:06<00:11, 11.11it/s][A
     41%|████      | 91/222 [00:06<00:11, 11.10it/s][A
     42%|████▏     | 93/222 [00:06<00:11, 11.16it/s][A
     43%|████▎     | 95/222 [00:07<00:11, 11.24it/s][A
     44%|████▎     | 97/222 [00:07<00:11, 11.23it/s][A
     45%|████▍     | 99/222 [00:07<00:11, 10.75it/s][A
     45%|████▌     | 101/222 [00:07<00:11, 10.92it/s][A
     46%|████▋     | 103/222 [00:07<00:10, 10.99it/s][A
     47%|████▋     | 105/222 [00:07<00:10, 11.11it/s][A
     48%|████▊     | 107/222 [00:08<00:10, 11.12it/s][A
     49%|████▉     | 109/222 [00:08<00:10, 11.18it/s][A
     50%|█████     | 111/222 [00:08<00:10, 11.09it/s][A
     51%|█████     | 113/222 [00:08<00:10, 10.89it/s][A
     52%|█████▏    | 115/222 [00:08<00:09, 10.98it/s][A
     53%|█████▎    | 117/222 [00:09<00:09, 11.07it/s][A
     54%|█████▎    | 119/222 [00:09<00:09, 11.20it/s][A
     55%|█████▍    | 121/222 [00:09<00:09, 11.22it/s][A
     55%|█████▌    | 123/222 [00:09<00:08, 11.25it/s][A
     56%|█████▋    | 125/222 [00:09<00:08, 11.31it/s][A
     57%|█████▋    | 127/222 [00:09<00:08, 11.14it/s][A
     58%|█████▊    | 129/222 [00:10<00:08, 11.27it/s][A
     59%|█████▉    | 131/222 [00:10<00:08, 11.19it/s][A
     60%|█████▉    | 133/222 [00:10<00:08, 10.75it/s][A
     61%|██████    | 135/222 [00:10<00:08, 10.48it/s][A
     62%|██████▏   | 137/222 [00:10<00:08, 10.60it/s][A
     63%|██████▎   | 139/222 [00:11<00:07, 10.68it/s][A
     64%|██████▎   | 141/222 [00:11<00:07, 10.72it/s][A
     64%|██████▍   | 143/222 [00:11<00:07, 10.82it/s][A
     65%|██████▌   | 145/222 [00:11<00:07, 10.69it/s][A
     66%|██████▌   | 147/222 [00:11<00:06, 10.82it/s][A
     67%|██████▋   | 149/222 [00:12<00:06, 10.62it/s][A
     68%|██████▊   | 151/222 [00:12<00:06, 10.86it/s][A
     69%|██████▉   | 153/222 [00:12<00:06, 10.96it/s][A
     70%|██████▉   | 155/222 [00:12<00:06, 11.02it/s][A
     71%|███████   | 157/222 [00:12<00:05, 11.03it/s][A
     72%|███████▏  | 159/222 [00:12<00:05, 10.97it/s][A
     73%|███████▎  | 161/222 [00:13<00:05, 11.00it/s][A
     73%|███████▎  | 163/222 [00:13<00:05, 10.59it/s][A
     74%|███████▍  | 165/222 [00:13<00:05, 10.59it/s][A
     75%|███████▌  | 167/222 [00:13<00:05, 10.77it/s][A
     76%|███████▌  | 169/222 [00:13<00:04, 10.87it/s][A
     77%|███████▋  | 171/222 [00:14<00:04, 11.00it/s][A
     78%|███████▊  | 173/222 [00:14<00:04, 11.06it/s][A
     79%|███████▉  | 175/222 [00:14<00:04, 11.09it/s][A
     80%|███████▉  | 177/222 [00:14<00:04, 11.00it/s][A
     81%|████████  | 179/222 [00:14<00:04, 10.73it/s][A
     82%|████████▏ | 181/222 [00:14<00:03, 10.77it/s][A
     82%|████████▏ | 183/222 [00:15<00:03, 10.84it/s][A
     83%|████████▎ | 185/222 [00:15<00:03, 10.93it/s][A
     84%|████████▍ | 187/222 [00:15<00:03, 10.89it/s][A
     85%|████████▌ | 189/222 [00:15<00:03, 10.92it/s][A
     86%|████████▌ | 191/222 [00:15<00:02, 11.02it/s][A
     87%|████████▋ | 193/222 [00:16<00:02, 11.06it/s][A
     88%|████████▊ | 195/222 [00:16<00:02, 10.96it/s][A
     89%|████████▊ | 197/222 [00:16<00:02, 10.96it/s][A
     90%|████████▉ | 199/222 [00:16<00:02, 10.66it/s][A
     91%|█████████ | 201/222 [00:16<00:01, 10.82it/s][A
     91%|█████████▏| 203/222 [00:16<00:01, 10.97it/s][A
     92%|█████████▏| 205/222 [00:17<00:01, 10.95it/s][A
     93%|█████████▎| 207/222 [00:17<00:01, 11.00it/s][A
     94%|█████████▍| 209/222 [00:17<00:01, 10.93it/s][A
     95%|█████████▌| 211/222 [00:17<00:01, 10.70it/s][A
     96%|█████████▌| 213/222 [00:17<00:00, 10.82it/s][A
     97%|█████████▋| 215/222 [00:18<00:00, 11.00it/s][A
     98%|█████████▊| 217/222 [00:18<00:00, 11.05it/s][A
     99%|█████████▊| 219/222 [00:18<00:00, 11.00it/s][A
    100%|█████████▉| 221/222 [00:18<00:00, 11.04it/s][A
    [A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/solidWhiteRight.mp4 
    
    CPU times: user 7.67 s, sys: 238 ms, total: 7.91 s
    Wall time: 20.3 s


Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.


```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
```





<video width="960" height="540" controls>
  <source src="test_videos_output/solidWhiteRight.mp4">
</video>




## Improve the draw_lines() function

**At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**

**Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

Now for the one with the solid yellow lane on the left. This one's more tricky!


```python
yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
try:
    #clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
    clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
    yellow_clip = clip2.fl_image(process_image)
    %time yellow_clip.write_videofile(yellow_output, audio=False)
finally:
    clip2.reader.close();
```

    [MoviePy] >>>> Building video test_videos_output/solidYellowLeft.mp4
    [MoviePy] Writing video test_videos_output/solidYellowLeft.mp4


    
      0%|          | 0/682 [00:00<?, ?it/s][A
      0%|          | 2/682 [00:00<00:38, 17.57it/s][A
      1%|          | 5/682 [00:00<00:36, 18.59it/s][A
      1%|          | 8/682 [00:00<00:34, 19.48it/s][A
      1%|▏         | 10/682 [00:00<00:35, 19.15it/s][A
      2%|▏         | 13/682 [00:00<00:33, 19.68it/s][A
      2%|▏         | 16/682 [00:00<00:33, 20.15it/s][A
      3%|▎         | 18/682 [00:00<00:33, 20.06it/s][A
      3%|▎         | 21/682 [00:01<00:32, 20.40it/s][A
      4%|▎         | 24/682 [00:01<00:31, 20.75it/s][A
      4%|▍         | 27/682 [00:01<00:32, 20.13it/s][A
      4%|▍         | 30/682 [00:01<00:31, 20.40it/s][A
      5%|▍         | 33/682 [00:01<00:31, 20.81it/s][A
      5%|▌         | 36/682 [00:01<00:30, 20.97it/s][A
      6%|▌         | 39/682 [00:01<00:30, 21.10it/s][A
      6%|▌         | 42/682 [00:02<00:30, 20.84it/s][A
      7%|▋         | 45/682 [00:02<00:45, 14.07it/s][A
      7%|▋         | 47/682 [00:02<00:50, 12.65it/s][A
      7%|▋         | 49/682 [00:02<00:52, 12.14it/s][A
      7%|▋         | 51/682 [00:02<00:53, 11.78it/s][A
      8%|▊         | 53/682 [00:03<00:54, 11.51it/s][A
      8%|▊         | 55/682 [00:03<00:55, 11.33it/s][A
      8%|▊         | 57/682 [00:03<00:55, 11.34it/s][A
      9%|▊         | 59/682 [00:03<00:56, 10.96it/s][A
      9%|▉         | 61/682 [00:03<00:56, 10.94it/s][A
      9%|▉         | 63/682 [00:04<00:57, 10.83it/s][A
     10%|▉         | 65/682 [00:04<00:57, 10.82it/s][A
     10%|▉         | 67/682 [00:04<00:55, 11.00it/s][A
     10%|█         | 69/682 [00:04<00:57, 10.62it/s][A
     10%|█         | 71/682 [00:04<00:56, 10.78it/s][A
     11%|█         | 73/682 [00:04<00:56, 10.85it/s][A
     11%|█         | 75/682 [00:05<00:56, 10.82it/s][A
     11%|█▏        | 77/682 [00:05<00:55, 10.97it/s][A
     12%|█▏        | 79/682 [00:05<00:54, 11.06it/s][A
     12%|█▏        | 81/682 [00:05<00:54, 11.08it/s][A
     12%|█▏        | 83/682 [00:05<00:54, 11.07it/s][A
     12%|█▏        | 85/682 [00:06<00:54, 10.99it/s][A
     13%|█▎        | 87/682 [00:06<00:53, 11.07it/s][A
     13%|█▎        | 89/682 [00:06<00:54, 10.95it/s][A
     13%|█▎        | 91/682 [00:06<00:55, 10.70it/s][A
     14%|█▎        | 93/682 [00:06<00:54, 10.90it/s][A
     14%|█▍        | 95/682 [00:06<00:53, 10.99it/s][A
     14%|█▍        | 97/682 [00:07<00:53, 10.90it/s][A
     15%|█▍        | 99/682 [00:07<00:53, 10.98it/s][A
     15%|█▍        | 101/682 [00:07<00:53, 10.96it/s][A
     15%|█▌        | 103/682 [00:07<00:54, 10.56it/s][A
     15%|█▌        | 105/682 [00:07<00:54, 10.66it/s][A
     16%|█▌        | 107/682 [00:08<00:53, 10.72it/s][A
     16%|█▌        | 109/682 [00:08<00:53, 10.79it/s][A
     16%|█▋        | 111/682 [00:08<00:53, 10.71it/s][A
     17%|█▋        | 113/682 [00:08<00:52, 10.82it/s][A
     17%|█▋        | 115/682 [00:08<00:52, 10.83it/s][A
     17%|█▋        | 117/682 [00:09<00:52, 10.83it/s][A
     17%|█▋        | 119/682 [00:09<00:52, 10.72it/s][A
     18%|█▊        | 121/682 [00:09<00:52, 10.69it/s][A
     18%|█▊        | 123/682 [00:09<00:52, 10.65it/s][A
     18%|█▊        | 125/682 [00:09<00:51, 10.80it/s][A
     19%|█▊        | 127/682 [00:09<00:51, 10.84it/s][A
     19%|█▉        | 129/682 [00:10<00:51, 10.80it/s][A
     19%|█▉        | 131/682 [00:10<00:50, 10.88it/s][A
     20%|█▉        | 133/682 [00:10<00:50, 10.91it/s][A
     20%|█▉        | 135/682 [00:10<00:51, 10.66it/s][A
     20%|██        | 137/682 [00:10<00:50, 10.80it/s][A
     20%|██        | 139/682 [00:11<00:51, 10.62it/s][A
     21%|██        | 141/682 [00:11<00:50, 10.71it/s][A
     21%|██        | 143/682 [00:11<00:49, 10.80it/s][A
     21%|██▏       | 145/682 [00:11<00:49, 10.88it/s][A
     22%|██▏       | 147/682 [00:11<00:49, 10.90it/s][A
     22%|██▏       | 149/682 [00:12<00:48, 11.04it/s][A
     22%|██▏       | 151/682 [00:12<00:48, 11.04it/s][A
     22%|██▏       | 153/682 [00:12<00:47, 11.07it/s][A
     23%|██▎       | 155/682 [00:12<00:47, 11.03it/s][A
     23%|██▎       | 157/682 [00:12<00:47, 10.98it/s][A
     23%|██▎       | 159/682 [00:12<00:47, 11.13it/s][A
     24%|██▎       | 161/682 [00:13<00:46, 11.09it/s][A
     24%|██▍       | 163/682 [00:13<00:47, 10.86it/s][A
     24%|██▍       | 165/682 [00:13<00:46, 11.00it/s][A
     24%|██▍       | 167/682 [00:13<00:48, 10.72it/s][A
     25%|██▍       | 169/682 [00:13<00:47, 10.81it/s][A
     25%|██▌       | 171/682 [00:14<00:46, 10.88it/s][A
     25%|██▌       | 173/682 [00:14<00:46, 10.93it/s][A
     26%|██▌       | 175/682 [00:14<00:45, 11.05it/s][A
     26%|██▌       | 177/682 [00:14<00:45, 11.10it/s][A
     26%|██▌       | 179/682 [00:14<00:45, 11.09it/s][A
     27%|██▋       | 181/682 [00:14<00:45, 11.07it/s][A
     27%|██▋       | 183/682 [00:15<00:45, 11.07it/s][A
     27%|██▋       | 185/682 [00:15<00:44, 11.20it/s][A
     27%|██▋       | 187/682 [00:15<00:44, 11.24it/s][A
     28%|██▊       | 189/682 [00:15<00:43, 11.21it/s][A
     28%|██▊       | 191/682 [00:15<00:43, 11.18it/s][A
     28%|██▊       | 193/682 [00:16<00:45, 10.77it/s][A
     29%|██▊       | 195/682 [00:16<00:45, 10.80it/s][A
     29%|██▉       | 197/682 [00:16<00:44, 10.80it/s][A
     29%|██▉       | 199/682 [00:16<00:45, 10.63it/s][A
     29%|██▉       | 201/682 [00:16<00:44, 10.83it/s][A
     30%|██▉       | 203/682 [00:16<00:44, 10.84it/s][A
     30%|███       | 205/682 [00:17<00:44, 10.76it/s][A
     30%|███       | 207/682 [00:17<00:44, 10.79it/s][A
     31%|███       | 209/682 [00:17<00:43, 10.84it/s][A
     31%|███       | 211/682 [00:17<00:43, 10.90it/s][A
     31%|███       | 213/682 [00:17<00:42, 10.95it/s][A
     32%|███▏      | 215/682 [00:18<00:42, 11.02it/s][A
     32%|███▏      | 217/682 [00:18<00:41, 11.08it/s][A
     32%|███▏      | 219/682 [00:18<00:41, 11.10it/s][A
     32%|███▏      | 221/682 [00:18<00:41, 11.16it/s][A
     33%|███▎      | 223/682 [00:18<00:43, 10.62it/s][A
     33%|███▎      | 225/682 [00:18<00:42, 10.72it/s][A
     33%|███▎      | 227/682 [00:19<00:41, 10.84it/s][A
     34%|███▎      | 229/682 [00:19<00:41, 10.91it/s][A
     34%|███▍      | 231/682 [00:19<00:41, 10.97it/s][A
     34%|███▍      | 233/682 [00:19<00:40, 11.04it/s][A
     34%|███▍      | 235/682 [00:19<00:40, 11.08it/s][A
     35%|███▍      | 237/682 [00:20<00:40, 11.04it/s][A
     35%|███▌      | 239/682 [00:20<00:40, 11.00it/s][A
     35%|███▌      | 241/682 [00:20<00:39, 11.03it/s][A
     36%|███▌      | 243/682 [00:20<00:39, 11.06it/s][A
     36%|███▌      | 245/682 [00:20<00:39, 11.08it/s][A
     36%|███▌      | 247/682 [00:20<00:39, 10.90it/s][A
     37%|███▋      | 249/682 [00:21<00:40, 10.77it/s][A
     37%|███▋      | 251/682 [00:21<00:39, 10.85it/s][A
     37%|███▋      | 253/682 [00:21<00:39, 10.82it/s][A
     37%|███▋      | 255/682 [00:21<00:40, 10.61it/s][A
     38%|███▊      | 257/682 [00:21<00:39, 10.73it/s][A
     38%|███▊      | 259/682 [00:22<00:39, 10.82it/s][A
     38%|███▊      | 261/682 [00:22<00:38, 10.81it/s][A
     39%|███▊      | 263/682 [00:22<00:38, 10.89it/s][A
     39%|███▉      | 265/682 [00:22<00:38, 10.97it/s][A
     39%|███▉      | 267/682 [00:22<00:37, 11.02it/s][A
     39%|███▉      | 269/682 [00:22<00:37, 11.06it/s][A
     40%|███▉      | 271/682 [00:23<00:37, 11.04it/s][A
     40%|████      | 273/682 [00:23<00:36, 11.07it/s][A
     40%|████      | 275/682 [00:23<00:36, 11.19it/s][A
     41%|████      | 277/682 [00:23<00:36, 11.12it/s][A
     41%|████      | 279/682 [00:23<00:37, 10.86it/s][A
     41%|████      | 281/682 [00:24<00:36, 10.95it/s][A
     41%|████▏     | 283/682 [00:24<00:36, 10.95it/s][A
     42%|████▏     | 285/682 [00:24<00:36, 10.94it/s][A
     42%|████▏     | 287/682 [00:24<00:35, 10.99it/s][A
     42%|████▏     | 289/682 [00:24<00:36, 10.78it/s][A
     43%|████▎     | 291/682 [00:24<00:36, 10.59it/s][A
     43%|████▎     | 293/682 [00:25<00:36, 10.75it/s][A
     43%|████▎     | 295/682 [00:25<00:34, 11.27it/s][A
     44%|████▎     | 297/682 [00:25<00:34, 11.30it/s][A
     44%|████▍     | 299/682 [00:25<00:34, 11.19it/s][A
     44%|████▍     | 301/682 [00:25<00:34, 11.04it/s][A
     44%|████▍     | 303/682 [00:26<00:34, 11.00it/s][A
     45%|████▍     | 305/682 [00:26<00:34, 10.93it/s][A
     45%|████▌     | 307/682 [00:26<00:34, 10.83it/s][A
     45%|████▌     | 309/682 [00:26<00:34, 10.79it/s][A
     46%|████▌     | 311/682 [00:26<00:34, 10.67it/s][A
     46%|████▌     | 313/682 [00:26<00:34, 10.76it/s][A
     46%|████▌     | 315/682 [00:27<00:34, 10.59it/s][A
     46%|████▋     | 317/682 [00:27<00:34, 10.62it/s][A
     47%|████▋     | 319/682 [00:27<00:33, 10.72it/s][A
     47%|████▋     | 321/682 [00:27<00:33, 10.83it/s][A
     47%|████▋     | 323/682 [00:27<00:32, 10.95it/s][A
     48%|████▊     | 325/682 [00:28<00:36,  9.85it/s][A
     48%|████▊     | 327/682 [00:28<00:34, 10.18it/s][A
     48%|████▊     | 329/682 [00:28<00:33, 10.43it/s][A
     49%|████▊     | 331/682 [00:28<00:33, 10.45it/s][A
     49%|████▉     | 333/682 [00:28<00:32, 10.60it/s][A
     49%|████▉     | 335/682 [00:29<00:32, 10.71it/s][A
     49%|████▉     | 337/682 [00:29<00:31, 10.85it/s][A
     50%|████▉     | 339/682 [00:29<00:31, 10.93it/s][A
     50%|█████     | 341/682 [00:29<00:30, 11.11it/s][A
     50%|█████     | 343/682 [00:29<00:31, 10.60it/s][A
     51%|█████     | 345/682 [00:30<00:31, 10.56it/s][A
     51%|█████     | 347/682 [00:30<00:30, 10.82it/s][A
     51%|█████     | 349/682 [00:30<00:30, 10.86it/s][A
     51%|█████▏    | 351/682 [00:30<00:32, 10.13it/s][A
     52%|█████▏    | 353/682 [00:30<00:34,  9.41it/s][A
     52%|█████▏    | 354/682 [00:30<00:34,  9.50it/s][A
     52%|█████▏    | 356/682 [00:31<00:32,  9.91it/s][A
     52%|█████▏    | 358/682 [00:31<00:31, 10.18it/s][A
     53%|█████▎    | 360/682 [00:31<00:30, 10.41it/s][A
     53%|█████▎    | 362/682 [00:31<00:30, 10.51it/s][A
     53%|█████▎    | 364/682 [00:31<00:29, 10.60it/s][A
     54%|█████▎    | 366/682 [00:32<00:29, 10.86it/s][A
     54%|█████▍    | 368/682 [00:32<00:29, 10.66it/s][A
     54%|█████▍    | 370/682 [00:32<00:28, 10.87it/s][A
     55%|█████▍    | 372/682 [00:32<00:28, 10.94it/s][A
     55%|█████▍    | 374/682 [00:32<00:28, 10.76it/s][A
     55%|█████▌    | 376/682 [00:32<00:28, 10.87it/s][A
     55%|█████▌    | 378/682 [00:33<00:27, 10.92it/s][A
     56%|█████▌    | 380/682 [00:33<00:27, 10.80it/s][A
     56%|█████▌    | 382/682 [00:33<00:27, 10.98it/s][A
     56%|█████▋    | 384/682 [00:33<00:27, 10.96it/s][A
     57%|█████▋    | 386/682 [00:33<00:27, 10.91it/s][A
     57%|█████▋    | 388/682 [00:34<00:27, 10.71it/s][A
     57%|█████▋    | 390/682 [00:34<00:26, 10.90it/s][A
     57%|█████▋    | 392/682 [00:34<00:26, 10.82it/s][A
     58%|█████▊    | 394/682 [00:34<00:26, 10.83it/s][A
     58%|█████▊    | 396/682 [00:34<00:26, 10.72it/s][A
     58%|█████▊    | 398/682 [00:34<00:26, 10.90it/s][A
     59%|█████▊    | 400/682 [00:35<00:26, 10.80it/s][A
     59%|█████▉    | 402/682 [00:35<00:25, 11.00it/s][A
     59%|█████▉    | 404/682 [00:35<00:26, 10.39it/s][A
     60%|█████▉    | 406/682 [00:35<00:25, 10.78it/s][A
     60%|█████▉    | 408/682 [00:35<00:25, 10.74it/s][A
     60%|██████    | 410/682 [00:36<00:25, 10.80it/s][A
     60%|██████    | 412/682 [00:36<00:24, 10.83it/s][A
     61%|██████    | 414/682 [00:36<00:24, 10.96it/s][A
     61%|██████    | 416/682 [00:36<00:24, 10.71it/s][A
     61%|██████▏   | 418/682 [00:36<00:24, 10.96it/s][A
     62%|██████▏   | 420/682 [00:37<00:24, 10.74it/s][A
     62%|██████▏   | 422/682 [00:37<00:24, 10.83it/s][A
     62%|██████▏   | 424/682 [00:37<00:23, 10.93it/s][A
     62%|██████▏   | 426/682 [00:37<00:23, 10.99it/s][A
     63%|██████▎   | 428/682 [00:37<00:23, 11.01it/s][A
     63%|██████▎   | 430/682 [00:37<00:23, 10.74it/s][A
     63%|██████▎   | 432/682 [00:38<00:23, 10.73it/s][A
     64%|██████▎   | 434/682 [00:38<00:23, 10.68it/s][A
     64%|██████▍   | 436/682 [00:38<00:22, 10.73it/s][A
     64%|██████▍   | 438/682 [00:38<00:22, 10.90it/s][A
     65%|██████▍   | 440/682 [00:38<00:22, 10.89it/s][A
     65%|██████▍   | 442/682 [00:39<00:21, 11.05it/s][A
     65%|██████▌   | 444/682 [00:39<00:22, 10.71it/s][A
     65%|██████▌   | 446/682 [00:39<00:22, 10.71it/s][A
     66%|██████▌   | 448/682 [00:39<00:21, 10.97it/s][A
     66%|██████▌   | 450/682 [00:39<00:20, 11.05it/s][A
     66%|██████▋   | 452/682 [00:39<00:21, 10.83it/s][A
     67%|██████▋   | 454/682 [00:40<00:20, 10.91it/s][A
     67%|██████▋   | 456/682 [00:40<00:20, 11.09it/s][A
     67%|██████▋   | 458/682 [00:40<00:20, 10.90it/s][A
     67%|██████▋   | 460/682 [00:40<00:20, 10.80it/s][A
     68%|██████▊   | 462/682 [00:40<00:20, 10.82it/s][A
     68%|██████▊   | 464/682 [00:41<00:20, 10.44it/s][A
     68%|██████▊   | 466/682 [00:41<00:20, 10.72it/s][A
     69%|██████▊   | 468/682 [00:41<00:19, 10.89it/s][A
     69%|██████▉   | 470/682 [00:41<00:19, 10.95it/s][A
     69%|██████▉   | 472/682 [00:41<00:19, 11.03it/s][A
     70%|██████▉   | 474/682 [00:41<00:18, 11.14it/s][A
     70%|██████▉   | 476/682 [00:42<00:18, 11.03it/s][A
     70%|███████   | 478/682 [00:42<00:18, 11.13it/s][A
     70%|███████   | 480/682 [00:42<00:18, 10.97it/s][A
     71%|███████   | 482/682 [00:42<00:18, 10.90it/s][A
     71%|███████   | 484/682 [00:42<00:17, 11.17it/s][A
     71%|███████▏  | 486/682 [00:43<00:17, 11.04it/s][A
     72%|███████▏  | 488/682 [00:43<00:17, 11.11it/s][A
     72%|███████▏  | 490/682 [00:43<00:17, 11.01it/s][A
     72%|███████▏  | 492/682 [00:43<00:17, 10.62it/s][A
     72%|███████▏  | 494/682 [00:43<00:17, 10.62it/s][A
     73%|███████▎  | 496/682 [00:44<00:17, 10.80it/s][A
     73%|███████▎  | 498/682 [00:44<00:16, 10.91it/s][A
     73%|███████▎  | 500/682 [00:44<00:17, 10.67it/s][A
     74%|███████▎  | 502/682 [00:44<00:16, 10.85it/s][A
     74%|███████▍  | 504/682 [00:44<00:16, 10.68it/s][A
     74%|███████▍  | 506/682 [00:44<00:16, 10.83it/s][A
     74%|███████▍  | 508/682 [00:45<00:16, 10.84it/s][A
     75%|███████▍  | 510/682 [00:45<00:15, 10.86it/s][A
     75%|███████▌  | 512/682 [00:45<00:15, 10.68it/s][A
     75%|███████▌  | 514/682 [00:45<00:15, 10.77it/s][A
     76%|███████▌  | 516/682 [00:45<00:15, 10.82it/s][A
     76%|███████▌  | 518/682 [00:46<00:15, 10.78it/s][A
     76%|███████▌  | 520/682 [00:46<00:15, 10.20it/s][A
     77%|███████▋  | 522/682 [00:46<00:15, 10.41it/s][A
     77%|███████▋  | 524/682 [00:46<00:15, 10.43it/s][A
     77%|███████▋  | 526/682 [00:46<00:14, 10.59it/s][A
     77%|███████▋  | 528/682 [00:47<00:14, 10.55it/s][A
     78%|███████▊  | 530/682 [00:47<00:14, 10.55it/s][A
     78%|███████▊  | 532/682 [00:47<00:14, 10.67it/s][A
     78%|███████▊  | 534/682 [00:47<00:13, 10.76it/s][A
     79%|███████▊  | 536/682 [00:47<00:13, 10.80it/s][A
     79%|███████▉  | 538/682 [00:47<00:13, 10.87it/s][A
     79%|███████▉  | 540/682 [00:48<00:13, 10.87it/s][A
     79%|███████▉  | 542/682 [00:48<00:12, 10.79it/s][A
     80%|███████▉  | 544/682 [00:48<00:12, 11.50it/s][A
     80%|████████  | 546/682 [00:48<00:11, 11.77it/s][A
     80%|████████  | 548/682 [00:48<00:11, 11.46it/s][A
     81%|████████  | 550/682 [00:48<00:11, 11.22it/s][A
     81%|████████  | 552/682 [00:49<00:11, 11.03it/s][A
     81%|████████  | 554/682 [00:49<00:11, 10.83it/s][A
     82%|████████▏ | 556/682 [00:49<00:11, 10.75it/s][A
     82%|████████▏ | 558/682 [00:49<00:11, 10.70it/s][A
     82%|████████▏ | 560/682 [00:49<00:11, 10.57it/s][A
     82%|████████▏ | 562/682 [00:50<00:11, 10.46it/s][A
     83%|████████▎ | 564/682 [00:50<00:11, 10.47it/s][A
     83%|████████▎ | 566/682 [00:50<00:10, 10.61it/s][A
     83%|████████▎ | 568/682 [00:50<00:10, 10.62it/s][A
     84%|████████▎ | 570/682 [00:50<00:10, 10.56it/s][A
     84%|████████▍ | 572/682 [00:51<00:10, 10.57it/s][A
     84%|████████▍ | 574/682 [00:51<00:10, 10.52it/s][A
     84%|████████▍ | 576/682 [00:51<00:10, 10.39it/s][A
     85%|████████▍ | 578/682 [00:51<00:09, 10.53it/s][A
     85%|████████▌ | 580/682 [00:51<00:09, 10.67it/s][A
     85%|████████▌ | 582/682 [00:52<00:09, 10.90it/s][A
     86%|████████▌ | 584/682 [00:52<00:09, 10.25it/s][A
     86%|████████▌ | 586/682 [00:52<00:09, 10.50it/s][A
     86%|████████▌ | 588/682 [00:52<00:08, 10.65it/s][A
     87%|████████▋ | 590/682 [00:52<00:08, 10.74it/s][A
     87%|████████▋ | 592/682 [00:52<00:08, 10.90it/s][A
     87%|████████▋ | 594/682 [00:53<00:07, 11.01it/s][A
     87%|████████▋ | 596/682 [00:53<00:07, 11.10it/s][A
     88%|████████▊ | 598/682 [00:53<00:07, 11.26it/s][A
     88%|████████▊ | 600/682 [00:53<00:07, 11.02it/s][A
     88%|████████▊ | 602/682 [00:53<00:07, 11.18it/s][A
     89%|████████▊ | 604/682 [00:54<00:07, 11.05it/s][A
     89%|████████▉ | 606/682 [00:54<00:06, 11.14it/s][A
     89%|████████▉ | 608/682 [00:54<00:06, 11.10it/s][A
     89%|████████▉ | 610/682 [00:54<00:06, 11.09it/s][A
     90%|████████▉ | 612/682 [00:54<00:06, 11.03it/s][A
     90%|█████████ | 614/682 [00:55<00:06,  9.87it/s][A
     90%|█████████ | 616/682 [00:55<00:06, 10.23it/s][A
     91%|█████████ | 618/682 [00:55<00:06, 10.47it/s][A
     91%|█████████ | 620/682 [00:55<00:05, 10.69it/s][A
     91%|█████████ | 622/682 [00:55<00:05, 10.80it/s][A
     91%|█████████▏| 624/682 [00:55<00:05, 10.87it/s][A
     92%|█████████▏| 626/682 [00:56<00:05, 10.88it/s][A
     92%|█████████▏| 628/682 [00:56<00:04, 10.87it/s][A
     92%|█████████▏| 630/682 [00:56<00:04, 10.87it/s][A
     93%|█████████▎| 632/682 [00:56<00:04, 10.92it/s][A
     93%|█████████▎| 634/682 [00:56<00:04, 10.80it/s][A
     93%|█████████▎| 636/682 [00:57<00:04, 10.69it/s][A
     94%|█████████▎| 638/682 [00:57<00:04, 10.79it/s][A
     94%|█████████▍| 640/682 [00:57<00:03, 10.89it/s][A
     94%|█████████▍| 642/682 [00:57<00:03, 11.10it/s][A
     94%|█████████▍| 644/682 [00:57<00:03, 10.87it/s][A
     95%|█████████▍| 646/682 [00:57<00:03, 10.96it/s][A
     95%|█████████▌| 648/682 [00:58<00:03, 10.87it/s][A
     95%|█████████▌| 650/682 [00:58<00:02, 10.85it/s][A
     96%|█████████▌| 652/682 [00:58<00:02, 10.90it/s][A
     96%|█████████▌| 654/682 [00:58<00:02, 10.68it/s][A
     96%|█████████▌| 656/682 [00:58<00:02, 10.89it/s][A
     96%|█████████▋| 658/682 [00:59<00:02, 10.94it/s][A
     97%|█████████▋| 660/682 [00:59<00:02, 10.63it/s][A
     97%|█████████▋| 662/682 [00:59<00:01, 10.77it/s][A
     97%|█████████▋| 664/682 [00:59<00:01, 10.84it/s][A
     98%|█████████▊| 666/682 [00:59<00:01, 10.84it/s][A
     98%|█████████▊| 668/682 [00:59<00:01, 10.84it/s][A
     98%|█████████▊| 670/682 [01:00<00:01, 10.89it/s][A
     99%|█████████▊| 672/682 [01:00<00:00, 10.87it/s][A
     99%|█████████▉| 674/682 [01:00<00:00, 11.03it/s][A
     99%|█████████▉| 676/682 [01:00<00:00, 10.79it/s][A
     99%|█████████▉| 678/682 [01:00<00:00, 10.86it/s][A
    100%|█████████▉| 680/682 [01:01<00:00, 10.85it/s][A
    100%|█████████▉| 681/682 [01:01<00:00, 11.14it/s][A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/solidYellowLeft.mp4 
    
    CPU times: user 23.7 s, sys: 720 ms, total: 24.4 s
    Wall time: 1min 2s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))
```





<video width="960" height="540" controls>
  <source src="test_videos_output/solidYellowLeft.mp4">
</video>




## Writeup and Submission

If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.


## 2. Identify any shortcomings

1. It may or may not work in all possible conditions like lighting, environmental
2. It can not detect too curvy roads, we only ised 1st degree of extraplation
3. If the camera position is slightly chanaged, than these setup will not work.

## 3. Possible improvements

1. If camera's position is getting changed than we should imeediately start calibration and can finetunes our poly_mask parameters.
2. We can make our algo. more robust to get lane detection of too curvy roads.
3. In bad conditions, instead of looking for lanes we can take help of sorrounding objects and can estimate route, as our fnal goal is to detect route and to go on.
4. For curvy roads we can dynamically change region of interest parameters in order to constantly get selection.


## Optional Challenge

Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!


```python
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
try:
    clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
    #clip3 = VideoFileClip('test_videos/challenge.mp4')
    challenge_clip = clip3.fl_image(process_image)
    %time challenge_clip.write_videofile(challenge_output, audio=False)
finally:
    clip3.reader.close();
```

    [MoviePy] >>>> Building video test_videos_output/challenge.mp4
    [MoviePy] Writing video test_videos_output/challenge.mp4


    
      0%|          | 0/126 [00:00<?, ?it/s][A
      2%|▏         | 2/126 [00:00<00:11, 11.27it/s][A
      3%|▎         | 4/126 [00:00<00:10, 11.25it/s][A
      5%|▍         | 6/126 [00:00<00:10, 11.29it/s][A
      6%|▋         | 8/126 [00:00<00:10, 11.46it/s][A
      7%|▋         | 9/126 [00:00<00:10, 10.98it/s][A
      9%|▊         | 11/126 [00:00<00:10, 11.24it/s][A
     10%|█         | 13/126 [00:01<00:09, 11.36it/s][A
     12%|█▏        | 15/126 [00:01<00:09, 11.36it/s][A
     13%|█▎        | 17/126 [00:01<00:09, 11.52it/s][A
     15%|█▌        | 19/126 [00:01<00:09, 11.63it/s][A
     17%|█▋        | 21/126 [00:01<00:09, 11.63it/s][A
     18%|█▊        | 23/126 [00:01<00:08, 11.79it/s][A
     20%|█▉        | 25/126 [00:02<00:08, 11.74it/s][A
     21%|██▏       | 27/126 [00:02<00:08, 11.71it/s][A
     23%|██▎       | 29/126 [00:02<00:08, 11.68it/s][A
     25%|██▍       | 31/126 [00:02<00:08, 11.51it/s][A
     26%|██▌       | 33/126 [00:02<00:08, 11.62it/s][A
     28%|██▊       | 35/126 [00:03<00:07, 11.54it/s][A
     29%|██▉       | 37/126 [00:03<00:07, 11.65it/s][A
     31%|███       | 39/126 [00:03<00:07, 11.64it/s][A
     33%|███▎      | 41/126 [00:03<00:07, 11.56it/s][A
     34%|███▍      | 43/126 [00:03<00:09,  8.37it/s][A
     35%|███▍      | 44/126 [00:04<00:10,  7.78it/s][A
     36%|███▌      | 45/126 [00:04<00:12,  6.59it/s][A
     37%|███▋      | 46/126 [00:04<00:12,  6.42it/s][A
     37%|███▋      | 47/126 [00:04<00:12,  6.19it/s][A
     38%|███▊      | 48/126 [00:04<00:12,  6.07it/s][A
     39%|███▉      | 49/126 [00:04<00:12,  6.04it/s][A
     40%|███▉      | 50/126 [00:05<00:12,  6.01it/s][A
     40%|████      | 51/126 [00:05<00:12,  6.00it/s][A
     41%|████▏     | 52/126 [00:05<00:12,  6.01it/s][A
     42%|████▏     | 53/126 [00:05<00:12,  5.94it/s][A
     43%|████▎     | 54/126 [00:05<00:12,  5.95it/s][A
     44%|████▎     | 55/126 [00:05<00:12,  5.89it/s][A
     44%|████▍     | 56/126 [00:06<00:12,  5.82it/s][A
     45%|████▌     | 57/126 [00:06<00:11,  5.84it/s][A
     46%|████▌     | 58/126 [00:06<00:11,  5.76it/s][A
     47%|████▋     | 59/126 [00:06<00:11,  5.79it/s][A
     48%|████▊     | 60/126 [00:06<00:11,  5.73it/s][A
     48%|████▊     | 61/126 [00:07<00:11,  5.79it/s][A
     49%|████▉     | 62/126 [00:07<00:10,  5.85it/s][A
     50%|█████     | 63/126 [00:07<00:10,  5.91it/s][A
     51%|█████     | 64/126 [00:07<00:10,  5.83it/s][A
     52%|█████▏    | 65/126 [00:07<00:10,  5.78it/s][A
     52%|█████▏    | 66/126 [00:07<00:10,  5.82it/s][A
     53%|█████▎    | 67/126 [00:08<00:10,  5.73it/s][A
     54%|█████▍    | 68/126 [00:08<00:10,  5.75it/s][A
     55%|█████▍    | 69/126 [00:08<00:09,  5.72it/s][A
     56%|█████▌    | 70/126 [00:08<00:09,  5.63it/s][A
     56%|█████▋    | 71/126 [00:08<00:09,  5.56it/s][A
     57%|█████▋    | 72/126 [00:08<00:09,  5.56it/s][A
     58%|█████▊    | 73/126 [00:09<00:09,  5.60it/s][A
     59%|█████▊    | 74/126 [00:09<00:09,  5.62it/s][A
     60%|█████▉    | 75/126 [00:09<00:09,  5.62it/s][A
     60%|██████    | 76/126 [00:09<00:08,  5.63it/s][A
     61%|██████    | 77/126 [00:09<00:08,  5.63it/s][A
     62%|██████▏   | 78/126 [00:10<00:08,  5.68it/s][A
     63%|██████▎   | 79/126 [00:10<00:08,  5.67it/s][A
     63%|██████▎   | 80/126 [00:10<00:08,  5.68it/s][A
     64%|██████▍   | 81/126 [00:10<00:07,  5.64it/s][A
     65%|██████▌   | 82/126 [00:10<00:07,  5.67it/s][A
     66%|██████▌   | 83/126 [00:10<00:07,  5.61it/s][A
     67%|██████▋   | 84/126 [00:11<00:07,  5.61it/s][A
     67%|██████▋   | 85/126 [00:11<00:07,  5.63it/s][A
     68%|██████▊   | 86/126 [00:11<00:07,  5.60it/s][A
     69%|██████▉   | 87/126 [00:11<00:06,  5.62it/s][A
     70%|██████▉   | 88/126 [00:11<00:06,  5.68it/s][A
     71%|███████   | 89/126 [00:11<00:06,  5.65it/s][A
     71%|███████▏  | 90/126 [00:12<00:06,  5.59it/s][A
     72%|███████▏  | 91/126 [00:12<00:06,  5.59it/s][A
     73%|███████▎  | 92/126 [00:12<00:06,  5.52it/s][A
     74%|███████▍  | 93/126 [00:12<00:05,  5.58it/s][A
     75%|███████▍  | 94/126 [00:12<00:05,  5.64it/s][A
     75%|███████▌  | 95/126 [00:13<00:05,  5.60it/s][A
     76%|███████▌  | 96/126 [00:13<00:05,  5.62it/s][A
     77%|███████▋  | 97/126 [00:13<00:05,  5.57it/s][A
     78%|███████▊  | 98/126 [00:13<00:05,  5.52it/s][A
     79%|███████▊  | 99/126 [00:13<00:04,  5.51it/s][A
     79%|███████▉  | 100/126 [00:13<00:04,  5.50it/s][A
     80%|████████  | 101/126 [00:14<00:04,  5.47it/s][A
     81%|████████  | 102/126 [00:14<00:04,  5.39it/s][A
     82%|████████▏ | 103/126 [00:14<00:04,  5.54it/s][A
     83%|████████▎ | 104/126 [00:14<00:04,  5.37it/s][A
     83%|████████▎ | 105/126 [00:14<00:04,  5.21it/s][A
     84%|████████▍ | 106/126 [00:15<00:03,  5.34it/s][A
     85%|████████▍ | 107/126 [00:15<00:03,  5.24it/s][A
     86%|████████▌ | 108/126 [00:15<00:03,  5.08it/s][A
     87%|████████▋ | 109/126 [00:15<00:03,  5.19it/s][A
     87%|████████▋ | 110/126 [00:15<00:03,  5.05it/s][A
     88%|████████▊ | 111/126 [00:16<00:03,  4.96it/s][A
     89%|████████▉ | 112/126 [00:16<00:02,  4.78it/s][A
     90%|████████▉ | 113/126 [00:16<00:02,  4.90it/s][A
     90%|█████████ | 114/126 [00:16<00:02,  4.93it/s][A
     91%|█████████▏| 115/126 [00:16<00:02,  4.87it/s][A
     92%|█████████▏| 116/126 [00:17<00:02,  4.80it/s][A
     93%|█████████▎| 117/126 [00:17<00:01,  4.72it/s][A
     94%|█████████▎| 118/126 [00:17<00:01,  4.60it/s][A
     94%|█████████▍| 119/126 [00:17<00:01,  4.84it/s][A
     95%|█████████▌| 120/126 [00:18<00:01,  4.55it/s][A
     96%|█████████▌| 121/126 [00:18<00:01,  4.82it/s][A
     97%|█████████▋| 122/126 [00:18<00:00,  4.63it/s][A
     98%|█████████▊| 123/126 [00:18<00:00,  4.83it/s][A
     98%|█████████▊| 124/126 [00:18<00:00,  4.67it/s][A
     99%|█████████▉| 125/126 [00:19<00:00,  4.83it/s][A
    [A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/challenge.mp4 
    
    CPU times: user 8.06 s, sys: 198 ms, total: 8.26 s
    Wall time: 23.4 s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))
```





<video width="960" height="540" controls>
  <source src="test_videos_output/challenge.mp4">
</video>



