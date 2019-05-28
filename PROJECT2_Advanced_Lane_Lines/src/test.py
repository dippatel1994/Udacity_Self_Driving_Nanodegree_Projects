from preprocessing import Preprocessor;
from thresholding import color_n_edge_threshold;
import os;
import cv2;
import matplotlib.pyplot as plt;
import numpy as np;
from slidingwindowsearch import LaneDetector;

if __name__=='__main__':
    test_output_folder = '../test_output_folder/'
    test_folder = '../test_images/';
    if not os.path.exists(test_folder):
        raise Exception("Test Folder does not exists");
    preprocessor = Preprocessor();
    lanedetector = LaneDetector();
    for file_name in os.listdir(test_folder):
        img = cv2.imread(os.path.join(test_folder,file_name));
        #img = img[...,::-1]; # Converting from BGR to RGB

        
        pre_image = preprocessor.preprocess_image(img);

        #cv2.imwrite(test_output_folder+'3.jpg',cv2.cvtColor(pre_image,cv2.COLOR_RGB2BGR));

        thresh_binary_img,color_binary = color_n_edge_threshold(pre_image);
        left_lane_fit,right_lane_fit,vis_img = lanedetector.search_lanes(thresh_binary_img);
        img_h = img.shape[0]
        ploty = np.linspace(start=0, stop=img_h, num=img_h);
        left_fitx = left_lane_fit[0]*ploty**2 + left_lane_fit[1]*ploty + left_lane_fit[2]
        right_fitx = right_lane_fit[0]*ploty**2 + right_lane_fit[1]*ploty + right_lane_fit[2];
        #thresh_binary_img = preprocessor.inv_perspective_transform(thresh_binary_img);
        overlay = np.uint8(np.dstack((thresh_binary_img,np.zeros_like(thresh_binary_img),np.zeros_like(thresh_binary_img)))*255);
        print("Radius : ",lanedetector.get_road_radius())
        radius_txt = "Radius of Curvature is : {:0.2f} m".format(lanedetector.get_road_radius());
        print(radius_txt);
        
        #cv2.putText(img,"Hello",(100,100), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),3,cv2.LINE_AA);

        pts_left = np.array( [np.transpose(np.vstack((left_fitx,ploty)))])
        pts_right = np.array( [np.flipud(np.transpose(np.vstack((right_fitx,ploty))))])
        points = np.hstack((pts_left, pts_right))

        plt.subplot(131);
        plt.title("Original Image")
        plt.imshow(img[...,::-1]);

        plt.subplot(132);
        plt.title("Thresholded Binary Image")
        plt.imshow(overlay);
        # plt.imshow(color_binary);
        # thresh_binary_img = np.uint8(thresh_binary_img*255);
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        mid_fit = np.mean([left_lane_fit,right_lane_fit],axis=0);
        mid_fitx = mid_fit[0]*ploty**2 + mid_fit[1]*ploty + mid_fit[2]
        plt.plot(mid_fitx, ploty, color='yellow')


        plt.subplot(133);
        plt.title("Lanes Detected")
        
        # Generate x and y values for plotting
        
        

        # plt.plot(left_fitx, ploty, color='yellow')
        cv2.fillPoly(overlay, np.int_([points]), (0, 200, 0))
        overlay = preprocessor.inv_perspective_transform(overlay);
        print(img.shape,overlay.shape)
        out_img = cv2.addWeighted(img[...,::-1], 1, overlay, 0.3, 0)
        cv2.putText(out_img,radius_txt,(50,100), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),3,cv2.LINE_AA)
        plt.imshow(out_img);

        # plt.plot(right_fitx, ploty, color='yellow')
        # mid_fit = np.mean([left_lane_fit,right_lane_fit],axis=0);
        # mid_fitx = mid_fit[0]*ploty**2 + mid_fit[1]*ploty + mid_fit[2]
        # plt.plot(mid_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)

        #cv2.imwrite(test_output_folder+file_name+'_thresh_binary.jpg',thresh_binary_img);

        plt.show(); 