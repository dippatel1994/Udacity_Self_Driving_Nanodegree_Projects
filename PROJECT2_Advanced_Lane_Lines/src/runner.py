from moviepy.editor import VideoFileClip
import cv2;
import os;
from preprocessing import Preprocessor;
from thresholding import color_n_edge_threshold;
import matplotlib.pyplot as plt;
import numpy as np;
from slidingwindowsearch import LaneDetector;


preprocessor = Preprocessor();
lanedetector = LaneDetector();
frame_count=0;
ip_frames_folder = 'ip_frames'
op_frames_folder = 'op_frames'


def get_frames_from_video(path):
    vidcap = cv2.VideoCapture(path);
    frames=[];

    while vidcap.isOpened():
        ret,frame = vidcap.read();
        if ret==True:
            frames.append(frame);
        else:
            break;
    vidcap.release();

    frames = np.array(frames);
    return frames;

def save_video_from_imgs(imgs,path):
    pass;

def clear_frame_folders(ip_frames_folder,op_frames_folder):
    global frame_count;
    frame_count=0;
    if(os.path.exists(ip_frames_folder)):
        for file in os.listdir(ip_frames_folder):
            os.remove(os.path.join(ip_frames_folder,file));
        os.rmdir(ip_frames_folder);
    os.mkdir(ip_frames_folder);
    if(os.path.exists(op_frames_folder)):
        for file in os.listdir(op_frames_folder):
            os.remove(os.path.join(op_frames_folder,file));    
        os.rmdir(op_frames_folder);
    os.mkdir(op_frames_folder);

def process_video_frame(img):
    img = img[...,::-1]; # Convert RGB to BGR
    out_img,thresh_binary_img,vis_img = process_image(img);
    out_img = out_img[...,::-1]; # Convert BGR to RGB
    return out_img#,thresh_binary_img,vis_img;

def process_image(img):
    if(img.shape[2]==4): #4 channel image i.e Image with Alpha channel
        img=img[:,:,:3];

    undist_img = preprocessor.undist_image(img);
    pre_image = preprocessor.preprocess_image(img)

    thresh_binary_img, color_binary = color_n_edge_threshold(pre_image);
    
    # plt.figure("Threshold bin image")
    # plt.imshow(thresh_binary_img);
    # plt.show();

    left_lane_fit, right_lane_fit, vis_img = lanedetector.search_lanes(thresh_binary_img);
    img_h = img.shape[0]
    overlay = lanedetector.overlay_lanes_n_text(img_h,thresh_binary_img);
    overlay = preprocessor.inv_perspective_transform(overlay);
    out_img = cv2.addWeighted(undist_img[...,::-1], 1, overlay, 0.3, 0) # drawing overlay on undistorted image
    
    mr,lr,rr = lanedetector.get_road_radius()

    radius_txt = "Radius of Curvature is : {:0.2f} m".format(mr)
    pos = lanedetector.get_vehicle_pos_from_center();
    pos_txt = "Distance from center : {:0.2f} m".format(pos);
    # radius_txt = "left:{:0.2f} m ,right:{:0.2f} m".format(lr,rr);
    radius_txt_color = (255,255,255);
    pos_txt_color = (255,255,255);
    
    cv2.putText(out_img,radius_txt,(50,100), cv2.FONT_HERSHEY_SIMPLEX, 2,radius_txt_color,3,cv2.LINE_AA)
    cv2.putText(out_img,pos_txt,(50,200), cv2.FONT_HERSHEY_SIMPLEX, 2,pos_txt_color,3,cv2.LINE_AA)
    out_img = cv2.cvtColor(out_img,cv2.COLOR_BGR2RGB);

    return out_img,thresh_binary_img,vis_img;

def process_n_save_frames(image):
    global frame_count;
    plt.imsave(ip_frames_folder+"/"+str(frame_count)+".jpg",image);
    op_image,_,_ = process_video_frame(image);
    plt.imsave(op_frames_folder+"/"+str(frame_count)+".jpg",op_image);
    frame_count+=1;
    return op_image;

def run_on_video(path):
    output = 'test_videos_output/output.mp4'
    try:
        ip_frames_folder = 'ip_frames'
        op_frames_folder = 'op_frames'
        clear_frame_folders(ip_frames_folder,op_frames_folder);
        # clip2 = VideoFileClip(path).subclip(20,26);
        clip2 = VideoFileClip(path);
        print("Processing Video..");
        # yellow_clip = clip2.fl_image(process_n_save_frames);
        yellow_clip = clip2.fl_image(process_video_frame);
        yellow_clip.write_videofile(output, audio=False)
    finally:
        clip2.reader.close();


def test_on_frame():
    
    # test_folder = './ip_frames/';
    # file_name = '1.jpg'

    test_folder = '../test_images/';
    file_name = 'straight_lines1.jpg'

    if not os.path.exists(test_folder):
        raise Exception("Test Folder does not exists");

    img = cv2.imread(os.path.join(test_folder,file_name));
    out_img,bin_img,vis_img = process_image(img);
    plt.figure("final vis image");
    plt.imshow(vis_img);
    plt.show();

    plt.figure("final out_img");
    plt.imshow(out_img[...,::-1]);
    plt.show();

if __name__ =='__main__':
    # test_on_frame();
    run_on_video('../project_video.mp4')