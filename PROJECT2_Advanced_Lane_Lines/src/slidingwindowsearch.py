import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from preprocessing import Preprocessor
from thresholding import color_n_edge_threshold

# Defining Constant
NUM_WINDOWS = 9
COLOR_GREEN = (0, 255, 0)
DEG_POLY = 2
# Define conversions in x and y from pixels space to meters
LANE_WIDTH_IN_METERS = 3.7;
#NUM_PIXELS_BETWEEN_LANES_AFTER_PERSP_TRANSFORM = 820 # 1063-243 taken from perspective transformed image of straight lane lines
NUM_PIXELS_BETWEEN_LANES_AFTER_PERSP_TRANSFORM = 794 # 1050-256 taken from perspective transformed image of straight lane lines, considering the distance from mid of lanes is 3.7 m
YM_PER_PIX = 30/720  # meters per pixel in y dimension
XM_PER_PIX = LANE_WIDTH_IN_METERS/NUM_PIXELS_BETWEEN_LANES_AFTER_PERSP_TRANSFORM  # meters per pixel in x dimension
IMG_H = 720;
IMG_W = 1280;
LAST_N_SAMPLES_TO_CONSIDER = 10;
# is_first_pass = True


class LaneDetector:
    is_first_pass = True;
    mv_avg_left_lane_fits = [];
    mv_avg_right_lane_fits = [];
    
    def search_lanes(self, img, num_windows=NUM_WINDOWS):

        #global is_first_pass;

        # img : Binary Warped Image
        img_h = img.shape[0]
        img_w = img.shape[1]
        # print('w : ', img_w);
        # Start searching for lanes from the lower half of the image
        # Calculating the Histogram
        hist = np.sum(img[img_h//3:, :], axis=0)
        hist_mid = hist.shape[0]//2

        # plt.plot(hist);
        # plt.show();

        # finding the base point for starting of left and right lane
        leftx_base = np.argmax(hist[:hist_mid])
        rightx_base = hist_mid + np.argmax(hist[hist_mid:])

        # Defining Window height and width
        win_h = img_h//num_windows
        win_w = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Identifying all non-zero pixels in the image
        nz_img = img.nonzero()
        nzy = np.array(nz_img[0])
        nzx = np.array(nz_img[1])

        #print('nzy : ',nzy)

        leftx_current = leftx_base
        rightx_current = rightx_base

        #print(rightx_base);

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Defining output image for visualization
        # vis_img = np.dstack((img,img,img))*255;
        vis_img = np.uint8(np.dstack((img, img, img))*255);
        # plt.figure("Vis Image slide1")
        # plt.imshow(vis_img);
        # plt.show();
        if self.is_first_pass == True:
            # print("First Pass")
            # iterating over window one by one
            for win in range(num_windows):
                # Calcualting window co-ordiantes
                left_win_x_left = leftx_current-win_w
                left_win_x_right = leftx_current+win_w

                right_win_x_left = rightx_current-win_w
                right_win_x_right = rightx_current+win_w

                win_y_top = img_h - (win+1)*win_h
                win_y_bottom = img_h-win*win_h

                #print(vis_img.shape)

                # Drawing Rectangles around window for Visualization
                cv2.rectangle(vis_img, (left_win_x_left, win_y_top),(left_win_x_right, win_y_bottom), COLOR_GREEN, 2)
                cv2.rectangle(vis_img, (right_win_x_left, win_y_top),(right_win_x_right, win_y_bottom), COLOR_GREEN, 2)

                # Identifying the nonzero pixels in x and y within the window
                pixels_in_left_win = ((nzx >= left_win_x_left) & (nzx <= left_win_x_right) &
                                      (nzy >= win_y_top) & (nzy < win_y_bottom)).nonzero()[0]

                pixels_in_right_win = ((nzx >= right_win_x_left) & (nzx <= right_win_x_right) &
                                       (nzy >= win_y_top) & (nzy < win_y_bottom)).nonzero()[0]

                #print("pixels_in_left_win : ",pixels_in_left_win);

                # Appending these indices to the lists
                left_lane_inds.append(pixels_in_left_win)
                right_lane_inds.append(pixels_in_right_win)

                # recenter the window if needed
                if(len(pixels_in_left_win) > minpix):
                    leftx_current = np.int(np.mean(nzx[pixels_in_left_win]))
                if(len(pixels_in_right_win) > minpix):
                    rightx_current = np.int(np.mean(nzx[pixels_in_right_win]))
            self.is_first_pass = False
            # Concatenating the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        else:
            #--------------- Second Pass -----------------
            # Assume you now have a new warped binary image
            # from the next frame of video
            # It's now much easier to find line pixels
            # self.left_lane_fit = l;
            # self.right_lane_fit = r;

            left_lane_inds = ((nzx > (self.left_lane_fit[0]*(nzy**2) + self.left_lane_fit[1]*nzy +
                                      self.left_lane_fit[2] - win_w)) & (nzx < (self.left_lane_fit[0]*(nzy**2) +
                                                                                self.left_lane_fit[1]*nzy + self.left_lane_fit[2] + win_w))).nonzero()[0]

            right_lane_inds = ((nzx > (self.right_lane_fit[0]*(nzy**2) + self.right_lane_fit[1]*nzy +
                                       self.right_lane_fit[2] - win_w)) & (nzx < (self.right_lane_fit[0]*(nzy**2) +
                                                                                  self.right_lane_fit[1]*nzy + self.right_lane_fit[2] + win_w))).nonzero()[0]

        # print(left_lane_inds);
        # print(right_lane_inds);

        # Extracting left and right line pixel positions
        self.leftx = nzx[left_lane_inds]
        self.lefty = nzy[left_lane_inds]
        self.rightx = nzx[right_lane_inds]
        self.righty = nzy[right_lane_inds]

        # Fiting a second order polynomial to each
        self.left_lane_fit = np.polyfit(self.lefty, self.leftx, DEG_POLY)
        self.right_lane_fit = np.polyfit(self.righty, self.rightx, DEG_POLY)

        if(len(self.mv_avg_left_lane_fits)>LAST_N_SAMPLES_TO_CONSIDER):
            self.mv_avg_left_lane_fits = self.mv_avg_left_lane_fits[1:];
            self.mv_avg_right_lane_fits = self.mv_avg_right_lane_fits[1:];

        self.mv_avg_left_lane_fits.append(self.left_lane_fit);
        self.mv_avg_right_lane_fits.append(self.right_lane_fit);

        #taking average of last n frames
        self.left_lane_fit = np.mean(self.mv_avg_left_lane_fits,axis=0);
        self.right_lane_fit = np.mean(self.mv_avg_right_lane_fits,axis=0);

        self.left_lane_fit_real_world = np.polyfit(
            self.lefty*YM_PER_PIX, self.leftx*XM_PER_PIX, DEG_POLY)
        self.right_lane_fit_real_world = np.polyfit(
            self.righty*YM_PER_PIX, self.rightx*XM_PER_PIX, DEG_POLY)

        #---------- Visulization part ---------------

        # Generate x and y values for plotting
        ploty = np.linspace(start=0, stop=img_h, num=img_h)
        # left_fitx = np.polyval(ploty,self.left_lane_fit)
        # right_fitx = np.polyval(ploty,self.right_lane_fit)

        self.left_fitx = self.left_lane_fit[0]*ploty**2 + \
            self.left_lane_fit[1]*ploty + self.left_lane_fit[2]
        self.right_fitx = self.right_lane_fit[0]*ploty**2 + \
            self.right_lane_fit[1]*ploty + self.right_lane_fit[2]

        vis_img[nzy[left_lane_inds], nzx[left_lane_inds]] = [255, 0, 0]
        vis_img[nzy[right_lane_inds], nzx[right_lane_inds]] = [0, 0, 255]
        #cv2.fillPoly
        # plt.imshow(vis_img)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        # plt.show();

        return self.left_lane_fit, self.right_lane_fit, vis_img

    # def calc_poly(coeff,val):
    #     l = len(coeff);
    #     y = np.ndarray(shape=l);
    #     for i in range(0,l,-1):
    #         y.append(coeff[i]*val**(l-i));
    #     return y;

    def get_road_radius(self):
        left_radius = self.get_curvature(self.left_lane_fit_real_world, IMG_H)
        right_radius = self.get_curvature(self.right_lane_fit_real_world, IMG_H)
        radius = (left_radius+right_radius)/2.0
        return radius,left_radius,right_radius;

    def get_curvature(self, fit, y):
        A, B, C = fit
        curvature = ((1 + (2 * A * y * YM_PER_PIX + B)**2) ** 1.5) / np.absolute(2 * A);
        return curvature

    def get_vehicle_pos_from_left(self):
        img_center = IMG_W//2;
        #ind = np.argmax(self.left_fity);
        # lane_center = (self.left_fitx + self.right_fitx)//2;
        # print(self.left_fitx[IMG_H-1]);
        # print(img_center);
        postion = (img_center-self.left_fitx[IMG_H-1]) * XM_PER_PIX;
        return postion;

    def get_vehicle_pos_from_center(self):
        postion = self.get_vehicle_pos_from_left()-(LANE_WIDTH_IN_METERS/2);
        return postion;

    def overlay_lanes_n_text(self,img_h,thresh_binary_img):
        # img_h = img.shape[0];
        ploty = np.linspace(start=0, stop=img_h, num=img_h)
        left_fitx = self.left_lane_fit[0]*ploty**2 + \
            self.left_lane_fit[1]*ploty + self.left_lane_fit[2]
        right_fitx = self.right_lane_fit[0]*ploty**2 + \
            self.right_lane_fit[1]*ploty + self.right_lane_fit[2]
        overlay = np.uint8(np.dstack((thresh_binary_img, np.zeros_like(
            thresh_binary_img), np.zeros_like(thresh_binary_img)))*255)
        
        pts_left = np.array([np.transpose(np.vstack((left_fitx, ploty)))])
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack((right_fitx, ploty))))])
        points = np.hstack((pts_left, pts_right))
            
        cv2.fillPoly(overlay, np.int_([points]), (0, 200, 0));
        
        return overlay;

def test_search_lanes():
    # img_path = '../test_output_folder/bin_img.jpg'
    img_path = '../test_images/straight_lines1.jpg'
    img = cv2.imread(img_path)
    preprocessor = Preprocessor()
    pimg = preprocessor.preprocess_image(img)
    bimg,_ = color_n_edge_threshold(pimg);
    lanedetector = LaneDetector();
    
    # search_lanes(img,self.is_first_pass=False,l=l,r=r);

    # plt.subplot(121);
    # plt.title("Original Image")
    # plt.imshow(img[:,:,::-1]);

    # # preprocessed_img = preprocess_image(img);
    # plt.subplot(122);
    plt.title("Visulized Image")
    plt.imshow(bimg,cmap='gray');

    l, r, vis_img = lanedetector.search_lanes(bimg, is_first_pass=True)

    # plt.show();


if __name__ == '__main__':
    # test_search_lanes()
    pass
