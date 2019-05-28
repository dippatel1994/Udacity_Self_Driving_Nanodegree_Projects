# -----------------------Imports-------------------
import numpy as np;
import cv2;
import matplotlib.pyplot as plt;
import os;
import pickle;

#-------------Defining Constants---------------
NX = 9;
NY = 6;
CAMERA_CALIBRATION_DATA_FOLDER = './cam_calib_data/'
CHESSBOARD_IMG_FOLDER = '../camera_cal/'
DEBUG_MODE = True;

is_persp_trans_mat_present=False;
persp_trans_mat = None;

class Preprocessor:

    distortion_coeffs=None;
    camera_matrix=None;
    persp_trans_mat = None;
    inv_persp_trans_mat = None;
    #is_init = False;

    def __init__(self):
        self.camera_matrix,self.distortion_coeffs = self.calibrate_camera();
        self.persp_trans_mat,self.inv_persp_trans_mat  = self.get_perspective_transform_matrix();
        
    #-------------Defining Functions---------------
    def get_files_in_folder(self,folder_path):
        if os.path.exists(folder_path):
            for file_name in os.listdir(folder_path):
                #print(file_name)
                yield os.path.join(folder_path,file_name);

    def get_files_in_folder_test(self):
        CHESSBOARD_IMG_FOLDER = '../camera_cal/'
        chessboard_imgs = self.get_files_in_folder(CHESSBOARD_IMG_FOLDER);
        while 1:
            try : 
                print(next(chessboard_imgs))
            except StopIteration:
                print("---------Over--------");
                break;

    def calibrate_camera(self,nx=NX,ny=NY,imgs_folder_path=CHESSBOARD_IMG_FOLDER,folder_to_save_camera_calib=CAMERA_CALIBRATION_DATA_FOLDER):
        calibration_data_file = os.path.join(folder_to_save_camera_calib,'camera_calib_data.p')
        if(DEBUG_MODE): print("Camera Calibration Starts...")
        # If Camera has been calibrated then load the data from pickle file
        if(os.path.exists(calibration_data_file)):
            if(DEBUG_MODE): print("Loading data from pickle")
            with open(calibration_data_file,'rb') as pickle_file:
                cam_calib_data = pickle.load(pickle_file);
            return cam_calib_data['camera_matrix'],cam_calib_data['distortion_coeffs'];
        
        if(DEBUG_MODE): print("Calibrating Camera...")

        # Else do the calibration    
        obj_points_list=[]; # 3D world co-ordinates
        img_points_list=[]; # Image co-ordinates

        obj_points = np.zeros(shape=(nx*ny,3),dtype=np.float32);
        obj_points[:,:2]=np.mgrid[0:nx,0:ny].T.reshape(-1,2);

        img_gen = self.get_files_in_folder(imgs_folder_path);

        while 1:
            try:
                img_path = next(img_gen);
                #if(DEBUG_MODE): print('image path : ',img_path);
                img = cv2.imread(img_path);
                #if(DEBUG_MODE): print('type of image :',type(img),'shape of image : ',img.shape);
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
                gray_shape = gray.shape[::-1]
                # Find the chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, (NX, NY), None);
                #If Found
                if ret==True:
                    obj_points_list.append(obj_points);
                    img_points_list.append(corners);
            except StopIteration:
                break;

        ret,camera_matrix,distortion_coeffs,rvecs,tvecs = cv2.calibrateCamera(obj_points_list,img_points_list,gray_shape,None,None) ;
        
        # Saving Camera Calibration data to pickle file
        camera_calib_dict = {'camera_matrix':camera_matrix,'distortion_coeffs':distortion_coeffs};
        with open(calibration_data_file,'wb') as pickel_file:
            pickle.dump(camera_calib_dict,pickel_file);
        
        if(DEBUG_MODE): print("Saved data to pickle")

        return camera_matrix,distortion_coeffs;

    def undist_image(self,img):
        #self.camera_matrix,self.distortion_coeffs = calibrate_camera();
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB);
        undist_img = cv2.undistort(img,self.camera_matrix,self.distortion_coeffs,None,self.camera_matrix);
        undist_bgr_img=cv2.cvtColor(undist_img,cv2.COLOR_RGB2BGR);
        return undist_bgr_img;

    def undist_test(self):
        img_gen = self.get_files_in_folder(CHESSBOARD_IMG_FOLDER);
        img_path = next(img_gen);
        #print(img_path)
        img = cv2.imread(img_path);
        rgb_img = img[:,:,::-1];
        plt.subplot(121);
        plt.title("Original Image");
        plt.imshow(rgb_img)
        undist_img = undist_image(img);
        
        plt.subplot(122);
        plt.title("Undistorted Image");
        plt.imshow(undist_img);

        plt.show();

    def get_perspective_transform_matrix(self):
        # Define Source Corners
        src = np.float32([
        (257, 685),
        (1050, 685),
        (583, 460),
        (702, 460)
        ])
        # Define Destination Corners.
        dst = np.float32([
            (257, 685),
            (1050, 685),
            (257, 0),
            (1050, 0)
        ])
        self.persp_trans_mat = cv2.getPerspectiveTransform(src, dst);
        self.inv_persp_trans_mat = cv2.getPerspectiveTransform(dst, src);
        return self.persp_trans_mat,self.inv_persp_trans_mat;
    
    def perspective_transform(self,img):
        # persp_trans_mat_pickle = 'persp_trans_mat.p'
        # persp_trans_data_folder = "./persp_trans_data/"
        # file_path = os.path.join(persp_trans_data_folder,persp_trans_mat_pickle);
        # if os.path.exists(file_path):
        #     with open(file_path) as pickle_file:
        #         pickle.load()
        # Convertin shape to width x height
        img_size = img.shape[1::-1] # ::-1 reverse the list and 1:: start from 1st index to start # or img.shape[::-1][1:]
        # Warp the image using OpenCV warpPerspective()
        warped_img = cv2.warpPerspective(img, self.persp_trans_mat, img_size);

        return warped_img;
    
    def inv_perspective_transform(self,img):
        # persp_trans_mat_pickle = 'persp_trans_mat.p'
        # persp_trans_data_folder = "./persp_trans_data/"
        # file_path = os.path.join(persp_trans_data_folder,persp_trans_mat_pickle);
        # if os.path.exists(file_path):
        #     with open(file_path) as pickle_file:
        #         pickle.load()
        # Converting shape to width x height
        img_size = img.shape[1::-1] # ::-1 reverse the list and 1:: start from 1st index to start # or img.shape[::-1][1:]
        # Warp the image using OpenCV warpPerspective()
        unwarped_img = cv2.warpPerspective(img, self.inv_persp_trans_mat, img_size);

        return unwarped_img;

    def test_perspective_transform(self):
        img_path = '../test_images/straight_lines1.jpg'
        img = cv2.imread(img_path);

        # src = np.float32([
        # (257, 685),
        # (1050, 685),
        # (583, 460),
        # (702, 460)
        # ])

        # plt.subplot(131);
        # plt.title("Original Image")
        # plt.imshow(img[:,:,::-1]);
        # # plt.scatter(src[:,0],src[:,1],c ='r',s=20)
        # # plt.show();

        undist_img = self.undist_image(img);
        # plt.subplot(132);
        # plt.title("Undistorted Image")
        # plt.imshow(undist_img[:,:,::-1]);
        # # plt.show();

        persp_trans_img = self.perspective_transform(undist_img);
        # plt.subplot(133);
        plt.title("Perspective Transformed Image")
        plt.imshow(persp_trans_img[:,:,::-1]);
        plt.show();

    def preprocess_image(self,img):
        '''
        Returns an preprocessed BGR Image
        '''
        unidst_img = self.undist_image(img); # Keeping this as it is and overlaying text on undist_image in process_image itself
        persp_trans_img = self.perspective_transform(unidst_img);
        # cropping the image to remove dash of the car from the image
        persp_trans_img[675:,:,:] = [0,0,0]
        #persp_trans_img = persp_trans_img[:675,:,:]
        return persp_trans_img;
        
    def test_preprocess_img(self):
        img_path = '../test_images/test1.jpg'
        img = cv2.imread(img_path);

        plt.subplot(121);
        plt.title("Original Image")
        plt.imshow(img[:,:,::-1]);

        preprocessed_img = preprocess_image(img);
        plt.subplot(122);
        plt.title("Preprocessed Image")
        plt.imshow(preprocessed_img);

        plt.show();


if __name__=='__main__':
    print("--------Main Starts----------");
        
    #test_preprocess_img();
    preprocessor = Preprocessor();
    preprocessor.test_perspective_transform();

    print("--------Main Ends----------");




