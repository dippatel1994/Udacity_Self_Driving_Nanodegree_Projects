import numpy as np
import cv2
import matplotlib.pyplot as plt
import os;

def abs_sobel_thresh(img,orient='x',ksize=3,thresh=(0,255)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    # 2) Take the gradient in x and y separately
    if orient=='x':
        sobel = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=ksize);
    elif orient=='y':
        sobel = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=ksize);
    # 3) Take the absolute value of the x and y gradients
    abs_sobel = np.absolute(sobel);
    # 4) Create a binary mask where direction thresholds are met
    mask = (abs_sobel>=thresh[0]) & (abs_sobel<=thresh[1])
    # 6) Return this mask as your binary_output image
    grad_binary = np.zeros_like(abs_sobel) # Remove this line
    grad_binary[mask]=1;
    return grad_binary

def mag_thresh(img, ksize=3, thresh=(0, 255)):
    # print("called with ",mag_thresh)
    # print(" ksize : ",ksize)
    # 1) Convert to grayscale
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=ksize);
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=ksize);
    # 3) Calculate the magnitude 
    sobel_mag = np.sqrt(sobelx**2+sobely**2);
    #print(sobel_mag);
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8((sobel_mag/np.amax(sobel_mag))*255);
    #print(scaled_sobel);
    # 5) Create a binary mask where mag thresholds are met
    mask = (scaled_sobel >= thresh[0]) & (scaled_sobel <=thresh[1]);
    # 6) Return this mask as your binary_output image
    # to show binary images with cv2.imshow please use dtype as float , otherwise default min.max will taken as (0,255)
    # and whole image will appear black on the screen
    mag_binary = np.zeros_like(scaled_sobel,dtype=np.float32); 
    mag_binary[mask]=1; 
    #print(mag_binary)
    return mag_binary

def dir_threshold(img, ksize=15, thresh=(0, np.pi/2)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=ksize);
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=ksize);
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx);
    abs_sobely = np.absolute(sobely);
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    theta = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    mask = (theta>=thresh[0]) & (theta<=thresh[1])
    # 6) Return this mask as your binary_output image
    # dir_binary = np.zeros_like(theta,dtype=np.float32) # Remove this line
    # dir_binary[mask]=1;
    #Inverse thresholding
    dir_binary = np.ones_like(theta,dtype=np.float32) # Remove this line
    dir_binary[mask]=0;

    return dir_binary

def combine_edge_thresholding(img):
    gradx = abs_sobel_thresh(img, orient='x', ksize=3, thresh=(45, 250))
    grady = abs_sobel_thresh(img, orient='y', ksize=3, thresh=(120, 255))
    # mag_binary = mag_thresh(img, ksize=3, thresh=(30, 128))
    mag_binary = mag_thresh(img, ksize=3, thresh=(10, 128))
    #dir_binary = dir_threshold(img, ksize=15, thresh=(0.7, 1.3))
    dir_binary = dir_threshold(img, ksize=15, thresh=(0.2895, 1.57))
    
    combined = np.zeros_like(dir_binary);

    combined[((gradx == 1) | (grady == 1)) & ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined;

def hls_threshold(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hsl_img = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    # 2) Apply a threshold to the S channel
    s_channel = hsl_img[:,:,2]
    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(s_channel,dtype=np.float32) # placeholder line
    binary_output[(s_channel>thresh[0]) & (s_channel<=thresh[1])]=1
    return binary_output

def hue_threshold(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hsl_img = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    # 2) Apply a threshold to the S channel
    h_channel = hsl_img[:,:,0]
    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(h_channel,dtype=np.float32) # placeholder line
    binary_output[(h_channel>thresh[0]) & (h_channel<=thresh[1])]=1
    return binary_output

def test_hsl_threshold():
    img = cv2.imread('../test_images/straight_lines2.jpg');
    #img = img[...,::-1]; # Converting from BGR to RGB

    plt.subplot(121);
    plt.title("Original Image")
    plt.imshow(img[...,::-1]);

    hls_binary = hls_threshold(img, thresh=(90, 255))
    
    plt.subplot(122);
    plt.title("Thresholded Image")
    plt.imshow(hls_binary,cmap='gray');

    plt.show();

def test_hue_threshold():
    #img = cv2.imread('../test_images/straight_lines2.jpg');
    img = cv2.imread('../test_output_folder/1.jpg');
    #img = img[...,::-1]; # Converting from BGR to RGB

    plt.subplot(121);
    plt.title("Original Image")
    plt.imshow(img[...,::-1]);

    hls_binary = hue_threshold(img, thresh=(15, 30))
    
    plt.subplot(122);
    plt.title("Thresholded Image")
    plt.imshow(hls_binary,cmap='gray');

    plt.show();

def color_n_edge_threshold(img):

    edge_binary = combine_edge_thresholding(img);

    # Threshold Saturation color channel
    s_thresh_min = 90
    s_thresh_max = 255
    s_binary = hls_threshold(img,(s_thresh_min,s_thresh_max));

    # Threshold on hue color channel
    h_thresh_min = 15
    h_thresh_max = 30
    h_binary = hue_threshold(img,(h_thresh_min,h_thresh_max));

    hs_binary = s_binary*h_binary;

    color_binary = np.uint8(np.dstack((np.zeros_like(edge_binary), edge_binary, hs_binary)) * 255)
    # combined_binary = None;
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(edge_binary)
    combined_binary[(edge_binary == 1) | (hs_binary == 1)] = 1;
        
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors

    #color_binary = np.uint8(np.dstack((np.zeros_like(edge_binary), edge_binary, s_binary)) * 255)
    # # Combine the two binary thresholds
    # combined_binary = np.zeros_like(edge_binary)
    # combined_binary[(s_binary == 1) | (edge_binary == 1)] = 1;

    return combined_binary,color_binary;

def test_color_n_edge_threshold():
    test_folder = '../test_images/';
    # test_folder = '../test_output_folder/';
    if not os.path.exists(test_folder):
        raise Exception("Test Folder does not exists");
        return None,None;

    for file_name in os.listdir(test_folder):
        img = cv2.imread(os.path.join(test_folder,file_name));
        #img = img[...,::-1]; # Converting from BGR to RGB

        plt.subplot(131);
        plt.title("Original Image")
        plt.imshow(img[...,::-1]);

        thresh_binary_img,color_binary = color_n_edge_threshold(img);
        
        plt.subplot(132);
        plt.title("Thresholded Color Image")
        plt.imshow(color_binary);

        plt.subplot(133);
        plt.title("Thresholded Binary Image")
        plt.imshow(thresh_binary_img,cmap='gray');

        plt.show(); 

def test_edge_thresholding():
    img = cv2.imread('../test_images/test5.jpg');
    #img = img[...,::-1]; # Converting from BGR to RGB

    plt.subplot(121);
    plt.title("Original Image")
    plt.imshow(img[...,::-1]);

    edge_img = combine_edge_thresholding(img);
    
    plt.subplot(122);
    plt.title("Thresholded Image")
    plt.imshow(edge_img,cmap='gray');

    plt.show();

class Threshold_Tuner:

    def __init__(self,func_to_tune,*args):
        self.func_to_tune=func_to_tune;
        self.func_args = args
    
    def nothing(self,x):
        pass
    
    def tune_threshold_by_sliders(self,img,low=None,high=None):
        # Create a black image, a window
        slider_window = np.zeros((300,400), np.uint8)
        cv2.namedWindow('slider_window',cv2.WINDOW_NORMAL);
        cv2.namedWindow('Original Image',cv2.WINDOW_NORMAL);
        cv2.imshow('Original Image',img);
        print(type(img));
        magx_win = np.zeros(img.shape, np.uint8)
        cv2.namedWindow('output_window',cv2.WINDOW_NORMAL);

        # create trackbars for Min Max Threshold value
        cv2.createTrackbar('Min','slider_window',0,255,self.nothing)
        cv2.createTrackbar('Max','slider_window',0,255,self.nothing)
 
        
        while 1:
            cv2.imshow('slider_window',slider_window);
            k = cv2.waitKey(100) & 0xFF; # cv2.waitKey() is requirred to pause execution and to plot images using cv2.imshow
            if k==27: # checking for ESC key press
                cv2.destroyAllWindows(); # if ESC Pressed close all the windows
                break;
            minn = cv2.getTrackbarPos('Min','slider_window');
            maxx = cv2.getTrackbarPos('Max','slider_window');
            if low!=None and high!=None:
                minn = low+(minn/255)*high
                maxx = low+(maxx/255)*high
            print(minn," ",maxx);
            # Apply each of the thresholding functions
            #out_img  = mag_thresh(img, ksize=ksize, mag_thresh=(minn, maxx))
            out_img = self.func_to_tune(img,thresh=(minn, maxx));

            #cv2.imshow(img);
            cv2.imshow('output_window',out_img);

if __name__=='__main__':
    test_color_n_edge_threshold();
    # img = cv2.imread('../test_images/test5.jpg');
    # edge_thresh = combine_edge_thresholding(img);
    # plt.subplot(121);
    # plt.imshow(img[...,::-1]);

    # plt.subplot(122);
    # plt.imshow(edge_thresh,cmap='gray');
    # plt.show();

    # test_color_n_edge_threshold();
    '''args = 15

    #test_color_n_edge_threshold();

    test_folder = '../test_output_folder/';
    if not os.path.exists(test_folder):
        raise Exception("Test Folder does not exists");

    for file_name in os.listdir(test_folder):
        img = cv2.imread(os.path.join(test_folder,file_name));
        # plt.imshow(img);
        # plt.show();

        # tuner = Threshold_Tuner(hue_threshold ,args);
        # tuner.tune_threshold_by_sliders(img,low=0,high=255)

        tuner = Threshold_Tuner(mag_thresh,args);
        tuner.tune_threshold_by_sliders(img,low=0,high=255)

        # tuner = Threshold_Tuner(dir_threshold,args);    
        # tuner.tune_threshold_by_sliders(img,low=0,high=np.pi/2.0);
        '''

    pass