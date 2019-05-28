
import cv2;
import matplotlib.pyplot as plt;

img_path = '../test_images/test1.jpg'
img = cv2.imread(img_path,0);

cv2.putText(img,"Hello",(100,100), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),3,cv2.LINE_AA);

plt.imshow(img);
plt.show();


