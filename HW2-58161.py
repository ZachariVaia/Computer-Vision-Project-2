#Ζαχάρη Βάια
#ΑΜ 58161
#ΕΡΓΑΣΙΑ 2 ΟΡΑΣΗ ΥΠΟΛΟΓΙΣΤΩΝ
import numpy as np
import cv2
#sift = cv2.xfeatures2d_SIFT.create(400)
surf = cv2.xfeatures2d_SURF.create(400)
filename1 = 'rio-01.png'
filename2 = 'rio-02.png'
filename3 = 'rio-03.png'
filename4 = 'rio-04.png'
# filename1 = 'photo_01.png'
# filename2 = 'photo_02.png'
# filename3 = 'photo_03.png'
# filename4 = 'photo_04.png'
img_1 = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
img_2 = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)
img_3 = cv2.imread(filename3, cv2.IMREAD_GRAYSCALE)
img_4 = cv2.imread(filename4, cv2.IMREAD_GRAYSCALE)
def panorama(img1,img2):
    #good match

    def match(d1, d2):
        n1 = d1.shape[0]
        n2 = d2.shape[0]

        matches = []
        for i in range(n1):
            fv = d1[i, :]
            diff = d2 - fv
            diff = np.abs(diff)
            distances = np.sum(diff, axis=1)

            i2 = np.argmin(distances)
            mindist2 = distances[i2]

            matches.append(cv2.DMatch(i, i2, mindist2))

        return matches
    def best_matches(matches1_2, matches2_1):
        good =[]

        for i in matches1_2:
            for j in matches2_1:
                t1 = i.trainIdx
                q1 = i.queryIdx
                q2 = j.queryIdx
                t2 = j.trainIdx
                if (q1 == t2) and (q2 == t1):
                    good.append(i)

        return good
    # #keypoints of each image, sift
    # kp1 = sift.detect(img1)
    # kp2 = sift.detect(img2)
    # # descriptors of each keypoint of each image
    # desc1 = sift.compute(img1, kp1)
    # desc2 = sift.compute(img2, kp2)
    #keypoints of each image, surf
    kp1 = surf.detect(img1)
    kp2 = surf.detect(img2)
    # descriptors of each keypoint of each image
    desc1 = surf.compute(img1, kp1)
    desc2 = surf.compute(img2, kp2)


    #MATCHES OF
    #img1 - img2
    matches1_2 = match(desc1[1], desc2[1])
    #img2 - img1
    matches2_1 = match(desc2[1], desc1[1])
    good_matches = []
    #img1 - img2
    good_matches = best_matches(matches1_2, matches2_1)

    # dimg = cv2.drawMatches(img1, desc1[0], img2, desc2[0], good_matches, None)
    # cv2.namedWindow('main3')
    # cv2.imshow('main3', dimg)
    # cv2.waitKey(0)


    img_pt1 = []  # for homography
    img_pt2 = []  # for homography
    img_pt1 = np.array([kp1[x.queryIdx].pt for x in good_matches])
    img_pt2 = np.array([kp2[x.trainIdx].pt for x in good_matches])

    M, mask = cv2.findHomography(img_pt2, img_pt1, cv2.RANSAC)  #ομογραφια

    img5 = cv2.warpPerspective(img2, M, (img1.shape[1]+1500, img1.shape[0]+1000))
    img5[0: img1.shape[0], 0: img1.shape[1]] = img1

    pass
    return img5

panorama1 = panorama(img_1, img_2)
panorama2 = panorama(img_3, img_4)
def seperate(img,l):
        # Threshold of the binary image

    ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Binary Threshold', binary)

    imgrows, imgcols = binary.shape
    pt1 = (0, 0)
    pt3 = np.zeros((imgrows, 2))
    pt4 = np.zeros((imgrows, 2))

    output = []
    cnt_row = 0
    cnt_col = 0
    cnt2_row = 0
    cnt2_col = 0
    cnt_row0 = 0
    cnt_col0 = 0
    cnt2_row0 = 0
    cnt2_col0 = 0
    r = 0
    r2 = 0
    for row in range(0, imgrows):
        for col in range(0, imgcols):
            if (binary[row, col] == 255) and (binary[row - 1, col] == 0) and (row != 0):
                pt3[r, 0] = row
                pt3[r, 1] = col
                cnt_row = row - 1
                cnt_col = col
                r = r + 1
            elif (binary[row, col] == 255) and (binary[row - 1, col] == 0) and (row != 0) and (col == 0):
                pt3[r, 0] = row
                pt3[r, 1] = col
                cnt_row = row - 1
                cnt_col = cnt_col + 1
                r = r + 1
            elif (binary[row, col] == 255) and (binary[row , col + 1 ] == 0):
                pt3[r, 0] = row
                pt3[r, 1] = col
                cnt_row = row - 1
                cnt_col = cnt_col + 1
                r = r + 1


            if (cnt_row0 == 0 ):
             cnt_row0 = row - 1
             cnt_col0 = col


            if (binary[row, col] == 255) and (binary[row , col - 1] == 0) and (col != 0) :
                pt4[r2, 0] = row
                pt4[r2, 1] = col
                cnt2_row = row
                cnt2_col = col
                r2 = r2 + 1
                if (cnt2_row0 == 0):
                    cnt2_row0 = row
                    cnt2_col0 = col

            col = col + 1
        row = row + 1
        col = 0


    pt3 = pt3[0:r,0:cnt_col-cnt_col0-1]
    pt4 = pt4[0:r2,0:cnt2_col-cnt2_col0-1]
    if (l == 1):
        array1 = np.zeros(shape =(int(pt3[r-1, 0]), int(pt3[r-1, 1])))
        array1 = img[0:int(pt3[r-1, 0]),0:int(pt3[r-1, 1])].copy()
    elif(l == 2):
        array1 = np.zeros(shape =(int(pt4[r2-1, 0]), int(pt4[r2-1, 1])))
        array1 = img[0:int(pt4[r2-1, 0]),0 :int(pt4[r2-1, 1])].copy()
    elif (l == 3):
        array1 = np.zeros(shape=(int(pt3[0, 0]), int(pt3[0, 1])))
        array1 = img[0:int(pt3[r - 1, 0]), 0:int(pt3[r - 1, 1])].copy()
    elif (l == 4):
        array1 = np.zeros(shape=(int(pt4[r2-1, 0]), int(pt4[0, 1])))
        array1 = img[0:int(pt4[r2-1, 0]), 0:int(pt4[0, 1])].copy()


    return array1

pan1 = seperate(panorama1,3)
pan2 = seperate(panorama2,1)
panorama3 = panorama(pan1, pan2)


cv2.namedWindow('pan1', cv2.WINDOW_NORMAL)
cv2.imshow('pan1', pan1)
cv2.namedWindow('pan2', cv2.WINDOW_NORMAL)
cv2.imshow('pan2', pan2)
cv2.namedWindow('panorama3', cv2.WINDOW_NORMAL)
cv2.imshow('panorama3', panorama3)
cv2.waitKey(0)
