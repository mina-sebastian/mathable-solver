import os
import cv2 as cv
import numpy as np

def template_editor(img_hsv):
    boundary_lower = 0
    boundary_upper = img_hsv.shape[0]
    boundary_left = 0
    boundary_right = img_hsv.shape[1]
    trackbar_name = 'Template Editor'
    cv.namedWindow(trackbar_name)
    def nothing(x):
        pass
    cv.createTrackbar('Lower', trackbar_name, boundary_lower, boundary_upper, nothing)
    cv.createTrackbar('Upper', trackbar_name, boundary_upper, boundary_upper, nothing)
    cv.createTrackbar('Left', trackbar_name, boundary_left, boundary_right, nothing)
    cv.createTrackbar('Right', trackbar_name, boundary_right, boundary_right, nothing)

    while True:
        lower = cv.getTrackbarPos('Lower', trackbar_name)
        upper = cv.getTrackbarPos('Upper', trackbar_name)
        left = cv.getTrackbarPos('Left', trackbar_name)
        right = cv.getTrackbarPos('Right', trackbar_name)
        img = img_hsv[lower:upper, left:right]
        cv.imshow(trackbar_name, img)
        k = cv.waitKey(1) & 0xFF
        print(lower, upper, left, right)
        if k == 27:
            break
    cv.destroyAllWindows()
    return lower, upper, left, right

folder_path = './templates/patches/'
export_path = './templates/patches/'
if not os.path.exists(export_path):
    os.makedirs(export_path)
list_imgs = os.listdir(folder_path)
for file_name in list_imgs:
    if file_name.endswith('.jpg'):
        img = cv.imread(folder_path + file_name)
        # img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        lower, upper, left, right = template_editor(img)
        print(file_name, lower, upper, left, right)
        cv.imwrite(export_path + 'a' + file_name, img[lower:upper, left:right])
    