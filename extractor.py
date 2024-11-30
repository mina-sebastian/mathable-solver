from time import sleep
import cv2 as cv
import numpy as np
import os

from config import boxes, box_size, total_size, search_range, something_wrong_threshold


def show_image(title,image, no_wait=False, delay=100):
    image=cv.resize(image,(0,0),fx=0.3,fy=0.3)
    cv.imshow(title,image)
    if no_wait:
        cv.waitKey(delay)
    else:
        cv.waitKey(0)
        cv.destroyAllWindows()



def get_corners(image_hsv, lower_hsv, upper_hsv, max=False):
    thresh = cv.inRange(image_hsv, lower_hsv, upper_hsv)    

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv.erode(thresh, kernel)

    # show_image('image_thresholded',thresh)

    edges =  cv.Canny(thresh, 200,400)
    # show_image('edges',edges)
    contours, _ = cv.findContours(edges,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    possible_top_left = None
    possible_top_right = None
    possible_bottom_right = None
    possible_bottom_left = None

    minY = None
    maxY = None
    minX = None
    maxX = None

    for i in range(len(contours)):
        possibleMinX = None
        possibleMaxX = None
        possibleMinY = None
        possibleMaxY = None
        if(len(contours[i]) >3):
            for point in contours[i].squeeze():
                if max:
                    if possibleMinX is None or point[0] < possibleMinX:
                        possibleMinX = point[0]
                    if possibleMaxX is None or point[0] > possibleMaxX:
                        possibleMaxX = point[0]
                    if possibleMinY is None or point[1] < possibleMinY:
                        possibleMinY = point[1]
                    if possibleMaxY is None or point[1] > possibleMaxY:
                        possibleMaxY = point[1]
                else:
                    if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                        possible_top_left = point
                    if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1] :
                        possible_bottom_right = point
                    if possible_top_right is None or point[0] - point[1] > possible_top_right[0] - possible_top_right[1]:
                        possible_top_right = point
                    if possible_bottom_left is None or point[0] - point[1] < possible_bottom_left[0] - possible_bottom_left[1]:
                        possible_bottom_left = point
        if possibleMaxX and possibleMaxX and possibleMaxX - possibleMinX > 20:
            if minX is None or possibleMinX < minX:
                minX = possibleMinX
            if maxX is None or possibleMaxX > maxX:
                maxX = possibleMaxX
            if minY is None or possibleMinY < minY:
                minY = possibleMinY
            if maxY is None or possibleMaxY > maxY:
                maxY = possibleMaxY
    
    if max:
        possible_top_left = (minX, minY)
        possible_top_right = (maxX, minY)
        possible_bottom_right = (maxX, maxY)
        possible_bottom_left = (minX, maxY)

    # thresh_2 = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
    # cv.circle(thresh_2, tuple(possible_top_left), 5, (0, 0, 255), -1)
    # cv.circle(thresh_2, tuple(possible_top_right), 5, (0, 0, 255), -1)
    # cv.circle(thresh_2, tuple(possible_bottom_right), 5, (0, 0, 255), -1)
    # cv.circle(thresh_2, tuple(possible_bottom_left), 5, (0, 0, 255), -1)
    # show_image('image_thresholded',thresh_2)
    
    return possible_top_left, possible_top_right, possible_bottom_right, possible_bottom_left


def extrage_careu(image):
    image_m_blur = cv.medianBlur(image,3)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5) 
    image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, 0, 0)
    # show_image('image_sharpened',image_sharpened)

    image_hsv = cv.cvtColor(image_sharpened, cv.COLOR_BGR2HSV)

    top_left_1, top_right_1, bottom_right_1, bottom_left_1 = get_corners(image_hsv, (20, 100, 100), (30, 255, 255))
    # luam si patratelele goale de pe margine cu mentiunea ca se iau maximele si minimele coordonatelor
    top_left_2, top_right_2, bottom_right_2, bottom_left_2 = get_corners(image_hsv, (59, 11, 255), (151, 59, 255), True)

    if top_left_1 is not None and top_left_2 is not None and (abs(top_left_1[1] - top_left_2[1]) > something_wrong_threshold or abs(top_left_1[0] - top_left_2[0]) > something_wrong_threshold):
        top_left_1 = top_left_2
    if top_right_1 is not None and top_right_2 is not None and (abs(top_right_1[1] - top_right_2[1]) > something_wrong_threshold or abs(top_right_1[0] - top_right_2[0]) > something_wrong_threshold):
        top_right_1 = top_right_2
    if bottom_right_1 is not None and bottom_right_2 is not None and (abs(bottom_right_1[1] - bottom_right_2[1]) > something_wrong_threshold or abs(bottom_right_1[0] - bottom_right_2[0]) > something_wrong_threshold):
        bottom_right_1 = bottom_right_2
    if bottom_left_1 is not None and bottom_left_2 is not None and (abs(bottom_left_1[1] - bottom_left_2[1]) > something_wrong_threshold or abs(bottom_left_1[0] - bottom_left_2[0]) > something_wrong_threshold):
        bottom_left_1 = bottom_left_2

    # daca nu avem un colt, il completam cu celalalt
    if top_left_1 is None:
        top_left_1 = top_left_2
    if top_right_1 is None:
        top_right_1 = top_right_2
    if bottom_right_1 is None:
        bottom_right_1 = bottom_right_2
    if bottom_left_1 is None:
        bottom_left_1 = bottom_left_2

    width = total_size
    height = total_size

    puzzle = np.array([top_left_1, top_right_1, bottom_left_1, bottom_right_1], dtype="float32")
    destination = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype="float32")
    M = cv.getPerspectiveTransform(puzzle, destination)
    result_sharpened = cv.warpPerspective(image_sharpened, M, (width, height))
    result = cv.warpPerspective(image, M, (width, height))
    
    return result, result_sharpened

def getRangeByCoord(coord, search_range):
    vMin = -1 * search_range
    vMax = search_range

    minCoord = coord + vMin
    maxCoord = coord + vMax
    if minCoord < 0:
        vMin += -1 * minCoord
    if maxCoord > total_size:
        vMax -= maxCoord - total_size
    return range(vMin, vMax)


def getLines(result):
    gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(~gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    # show_image('trsh', thresh)

    horizontal = thresh.copy()
    horizontal_size = total_size // 5
    horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv.erode(horizontal, horizontal_structure)
    horizontal = cv.dilate(horizontal, horizontal_structure)
    # show_image('hrz', horizontal)

    vertical = thresh.copy()
    vertical_size = total_size // 5
    vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
    vertical = cv.erode(vertical, vertical_structure)
    vertical = cv.dilate(vertical, vertical_structure)
    # show_image('vrt', vertical)

    grid = cv.add(horizontal, vertical)
    # print(grid.shape)
    # print(grid)

    # show_image('grid', grid)

    lines_vertical = []
    lines_horizontal = []

    # vertical
    for coordX in range(0, total_size + 1, box_size):
        # cv.line(grid, line[0], line[1], (0, 255, 0), 5)
        votes = np.array([0] * (2 * search_range + 1))
        rg = getRangeByCoord(coordX, search_range)
        if coordX == 0:
            lines_vertical.append([(0, 0), (0, total_size)])
            continue
        if coordX == total_size:
            lines_vertical.append([(total_size, 0), (total_size, total_size)])
            continue
        for i in rg:
            for coordY in range(0, total_size):
                if grid[coordY, coordX + i] == 255:
                    votes[i + search_range] += 1
        
        max_idx = np.argmax(votes)
        if votes[max_idx] > 0:
            lines_vertical.append([(coordX + max_idx - search_range, 0), (coordX + max_idx - search_range, total_size)])

    # horizontal
    for coordY in range(0, total_size + 1, box_size):
        votes = np.array([0] * (2 * search_range + 1))
        rg = getRangeByCoord(coordY, search_range)
        if coordY == 0:
            lines_horizontal.append([(0, 0), (total_size, 0)])
            continue
        if coordY == total_size:
            lines_horizontal.append([(0, total_size), (total_size, total_size)])
            continue
        # print(rg)
        for i in rg:
            for coordX in range(0, total_size):
                if grid[coordY + i, coordX] == 255:
                    votes[i + search_range] += 1
        
        max_idx = np.argmax(votes)
        # print('votes')
        # print(votes)
        if votes[max_idx] > 0:
            lines_horizontal.append([(0, coordY + max_idx - search_range), (total_size, coordY + max_idx - search_range)])

    return lines_vertical, lines_horizontal


def find_color_values_using_trackbar(frame):
 
    def nothing(x):
        pass

    cv.namedWindow("Trackbar") 
    cv.createTrackbar("LH", "Trackbar", 0, 255, nothing)
    cv.createTrackbar("LS", "Trackbar", 0, 255, nothing)
    cv.createTrackbar("LV", "Trackbar", 0, 255, nothing)
    cv.createTrackbar("UH", "Trackbar", 255, 255, nothing)
    cv.createTrackbar("US", "Trackbar", 255, 255, nothing)
    cv.createTrackbar("UV", "Trackbar", 255, 255, nothing)
    
    
    while True:

        l_h = cv.getTrackbarPos("LH", "Trackbar")
        l_s = cv.getTrackbarPos("LS", "Trackbar")
        l_v = cv.getTrackbarPos("LV", "Trackbar")
        u_h = cv.getTrackbarPos("UH", "Trackbar")
        u_s = cv.getTrackbarPos("US", "Trackbar")
        u_v = cv.getTrackbarPos("UV", "Trackbar")


        l = np.array([l_h, l_s, l_v])
        u = np.array([u_h, u_s, u_v])
        mask_table_hsv = cv.inRange(frame, l, u)        

        res = cv.bitwise_and(frame, frame, mask=mask_table_hsv)    
        
        cv.imshow("Frame", cv.resize(frame, (780, 540), 
               interpolation = cv.INTER_LINEAR))
        cv.imshow("Mask", cv.resize(mask_table_hsv, (780, 540), 
               interpolation = cv.INTER_LINEAR))
        cv.imshow("Res", cv.resize(res, (780, 540), 
               interpolation = cv.INTER_LINEAR))

        if cv.waitKey(25) & 0xFF == ord('q'):
                break
    cv.destroyAllWindows()

def get_patches(image_hsv, lines_vertical, lines_horizontal):
    rez = np.zeros((boxes, boxes), dtype='object')
    for i in range(len(lines_horizontal)-1):
        for j in range(len(lines_vertical)-1):
            y_min = lines_vertical[j][0][0]
            y_max = lines_vertical[j + 1][1][0]
            x_min = lines_horizontal[i][0][1]
            x_max = lines_horizontal[i + 1][1][1]
            patch = image_hsv[x_min:x_max, y_min:y_max].copy()
            patch = cv.resize(patch, (box_size, box_size))
            # show_image('patch', patch, True, 100)
            rez[i, j] = patch
    return rez



def determina_configuratie_careu_ox(img_name, image_hsv, lines_horizontal, lines_vertical):
    matrix = np.empty((boxes, boxes), dtype='str')

    lower_hsv_rem_3x = (0, 0, 122)
    upper_hsv_rem_3x = (122, 255, 255)

    lower_hsv_box = (124, 162, 180)
    upper_hsv_box = (214, 255, 231)
    # show_image('image_hsv', image_hsv)
    # find_color_values_using_trackbar(image_hsv)
    thresh_3x = cv.inRange(image_hsv, lower_hsv_rem_3x, upper_hsv_rem_3x)
    # show_image('thresh_rem_3x', thresh_3x)

    thresh_boxes = cv.inRange(image_hsv, lower_hsv_box, upper_hsv_box)
    thresh_boxes = cv.erode(thresh_boxes, np.ones((3, 3), np.uint8))
    thresh_boxes = cv.dilate(thresh_boxes, np.ones((3, 3), np.uint8))
    show_image('thresh_boxes', thresh_boxes)
    for i in range(len(lines_horizontal)-1):
        for j in range(len(lines_vertical)-1):
            y_min = lines_vertical[j][0][0]
            y_max = lines_vertical[j + 1][1][0]
            x_min = lines_horizontal[i][0][1]
            x_max = lines_horizontal[i + 1][1][1]
            patch_mask_3x = thresh_3x[x_min:x_max, y_min:y_max].copy()
            mean_3x = np.mean(patch_mask_3x)
            # print(mean_3x)
            if mean_3x > 127 and (i == 0 or i == boxes - 1) and (j == 0 or j == boxes - 1):
                matrix[i, j] = ''
                continue
            # show_image('patch', patch_mask, True)
            patch_mask_box = thresh_boxes[x_min:x_max, y_min:y_max].copy()
            #completati codul aici
            mean_mask_box = np.mean(patch_mask_box)
            if mean_mask_box > 127:
                patch = image_hsv[x_min:x_max, y_min:y_max].copy()
                cv.imwrite(f'./tables/patches/{img_name}_{i}_{j}.jpg', patch)
                matrix[i, j] = '1'
                # show_image('patch', patch, True, 100)
            
    return matrix
if __name__ == '__main__':
    file_path = './antrenare/'
    for filename in os.listdir(file_path):
        if filename.endswith('.jpg'):
            img = cv.imread(file_path + filename)
            print(filename)
            result, result_sharpened = extrage_careu(img)
            show_image('result', result)
            
            lines_vertical, lines_horizontal = getLines(result)

            rez = determina_configuratie_careu_ox(filename, result, lines_horizontal, lines_vertical)
            # print(rez)

            # for line in lines_vertical:
            #     cv.line(result, line[0], line[1], (0, 255, 0), 5)
            # for line in lines_horizontal:
            #     cv.line(result, line[0], line[1], (0, 0, 255), 5)

            # result = cv.resize(result, (0, 0), fx=0.3, fy=0.3)
            for i in range(boxes):
                for j in range(boxes):
                    if rez[j, i] == '1':
                        cv.rectangle(result, (i * box_size, j * box_size), ((i + 1) * box_size, (j + 1) * box_size), (0, 0, 255), 2)
            show_image('Grid', result)

    # for filename in os.listdir('./tables/'):
    #     if filename.endswith('.jpg'):
    #         img = cv.imread('./tables/' + filename)
            
            
    #         lines_vertical, lines_horizontal = getLines(result)

    #         for line in lines_vertical:
    #             cv.line(result, line[0], line[1], (0, 255, 0), 5)
    #         for line in lines_horizontal:
    #             cv.line(result, line[0], line[1], (0, 0, 255), 5)
    #         show_image('Grid', result)

    # files = os.listdir('./tables/patches/')
    # x_nb = 98
    # y_nb = int(np.ceil(len(files) / x_nb))
    # # put the images in a collage
    # x = 0
    # y = 0
    # box_collage_size = 64
    # collage_final = np.zeros((box_collage_size * y_nb, box_collage_size * x_nb), dtype=np.uint8)
    # for patch in files:
    #     img = cv.imread(f'./tables/patches/{patch}')
    #     # img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    #     thresh = cv.inRange(img, (0, 0, 0), (255, 255, 100))

    #     image_m_blur = cv.medianBlur(thresh,3)
    #     image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5) 
    #     thresh = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8, 0)
        
    #     thresh = cv.erode(thresh, np.ones((3, 3), np.uint8))
    #     thresh = cv.dilate(thresh, np.ones((3, 3), np.uint8))

    #     horizontal = thresh.copy()
    #     horizontal_size = thresh.shape[1] // 1
    #     horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    #     horizontal = cv.erode(horizontal, horizontal_structure)
    #     horizontal = cv.dilate(horizontal, horizontal_structure)
    #     # show_image('hrz', horizontal)

    #     vertical = thresh.copy()
    #     vertical_size = thresh.shape[1] // 1
    #     vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
    #     vertical = cv.erode(vertical, vertical_structure)
    #     vertical = cv.dilate(vertical, vertical_structure)
    #     # show_image('vrt', vertical)

    #     grid = cv.add(horizontal, vertical)
    #     # remove the lines from the image
    #     thresh = cv.bitwise_and(thresh, cv.bitwise_not(grid))

    #     # thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
    #     #add with red the removed lines
    #     # print(thresh.shape)
    #     # thresh[horizontal > 0] = [0, 0, 255]
    #     # thresh[vertical > 0] = [0, 0, 255]

    #     contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    #     # get contours that have the center of mass close to the center of the image
    #     newImg = np.zeros_like(thresh)
    #     for contour in contours:
    #         M = cv.moments(contour)
    #         if M['m00'] == 0:
    #             continue
    #         cX = int(M['m10'] / M['m00'])
    #         cY = int(M['m01'] / M['m00'])
    #         if abs(cX - thresh.shape[1] // 2) < 100 and abs(cY - thresh.shape[0] // 2) < 100:
    #             cv.drawContours(newImg, [contour], -1, 255, 2)
    #             break
            
        

    #     # show_image('patch', thresh, True, 300)
    #     # find_color_values_using_trackbar(img)


    #     collage_final[y * box_collage_size:(y + 1) * box_collage_size, x * box_collage_size:(x + 1) * box_collage_size] = cv.resize(newImg, (box_collage_size, box_collage_size))
    #     x += 1
    #     if x == x_nb:
    #         x = 0
    #         y += 1
    # show_image('Collage', collage_final)


        
