import cv2
import os
import shutil

# Load the images
image1 = cv2.imread(r'simTest\querry.jpg')  
root = r'simTest\dataBase'

image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
minimum = 1000
for filename in os.listdir(root):
    image2 = cv2.imread(os.path.join(root,filename))
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

    # Calculate the histogram and normalize it
    hist_image1 = cv2.calcHist([image1], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_image1, hist_image1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
    hist_image2 = cv2.calcHist([image2], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_image2, hist_image2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);

    # find the metric value
    similarity = cv2.compareHist(hist_image1, hist_image2, cv2.HISTCMP_BHATTACHARYYA)
    if(similarity<minimum):
        minimum = similarity
        match_file = filename
print(match_file," : ", minimum)
shutil.copy(os.path.join(root,match_file),"matched_image")