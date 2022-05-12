import cv2
import numpy as np
import os
def PreProcessing(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Chuyển sang kênh màu xám để đặt threshhold

    ret,thresh = cv2.threshold(gray,127,255,0,cv2.THRESH_TOZERO_INV) #threshhold chia thàng 2 mảng với giá trị pixel >=127 và <127
    #các giá trị có giá trị pixel bé sẽ gán bằng 0 (đen) và các giá trị khác sẽ là rgb(255,255,255) (bước nhị phân hoá ảnh)

    # cv2.imshow('Thresholded original',thresh)# show threshhold image
    cv2.waitKey()
    ## Get contours
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("img",img)
    cv2.waitKey()
    #tập các điểm-liên-tục tạo thành một đường cong (curve) gọi là contours. Việc tìm contours giúp tìm ra các object có trong hình.
    #hàm trên sẽ lưu các đường biên( vector của các điểm) ở array contours, hierachy lưu giá trị đường viền

    ## only draw contour that have big areas
    imx = img.shape[0]
    imy = img.shape[1]
    lp_area = (imx * imy) / 10# tìm area có giá trị =1/10  ảnh




    ## Get only rectangles given exceeding area
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01 * cv2.arcLength(cnt, True), True)
        ## calculate number of vertices
        print(len(approx))

        if len(approx) == 4 and cv2.contourArea(cnt) > lp_area:
            tmp_img = img.copy()
            cv2.drawContours(tmp_img, [cnt], 0, (0, 255, 255), 6)
            cv2.imshow('Contour Borders', tmp_img)
            cv2.waitKey(0)

            tmp_img = img.copy()
            cv2.drawContours(tmp_img, [cnt], 0, (255, 0, 255), -1)
            # cv2.imshow('Contour Filled', tmp_img)
            cv2.waitKey(0)


            # Make a hull arround the contour and draw it on the original image
            tmp_img = img.copy()
            mask = np.zeros((img.shape[:2]), np.uint8)
            hull = cv2.convexHull(cnt)
            cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)
            # cv2.imshow('Convex Hull Mask', mask)
            cv2.waitKey(0)


            # Draw minimum area rectangle
            tmp_img = img.copy()
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(tmp_img, [box], 0, (0, 0, 255), 2)
            # cv2.imshow('Minimum Area Rectangle', tmp_img)
            cv2.waitKey(0)


            # Draw bounding rectangle
            tmp_img = img.copy()
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(tmp_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.imshow('Bounding Rectangle', tmp_img)
            cv2.waitKey(0)


            # Bounding Rectangle and Minimum Area Rectangle
            tmp_img = img.copy()
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(tmp_img, [box], 0, (0, 0, 255), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(tmp_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('Bounding Rectangle', tmp_img)
            cv2.waitKey(0)


            # determine the most extreme points along the contour
            # https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
            tmp_img = img.copy()
            extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
            extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
            extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
            extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
            cv2.drawContours(tmp_img, [cnt], -1, (0, 255, 255), 2)
            cv2.circle(tmp_img, extLeft, 8, (0, 0, 255), -1)
            cv2.circle(tmp_img, extRight, 8, (0, 255, 0), -1)
            cv2.circle(tmp_img, extTop, 8, (255, 0, 0), -1)
            cv2.circle(tmp_img, extBot, 8, (255, 255, 0), -1)


            print("Corner Points: ", extLeft, extRight, extTop, extBot)


            cv2.imshow('img contour drawn', tmp_img)
            cv2.waitKey(0)
            #cv2.destroyAllWindows()



            ## Perspective Transform
            tmp_img = img.copy()
            pts = np.array([extLeft, extRight, extTop, extBot])
            warped = four_point_transform(tmp_img, pts)
            img=correct_orientation(warped)
            path=r'C:\Users\nhatn\PycharmProjects\KTLTPython\image_result'
            cv2.imshow("Original", img)
            cv2.imwrite(os.path.join(path , 'Original.jpg'), img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
            bg = cv2.morphologyEx(img, cv2.MORPH_DILATE, se)
            out_gray = cv2.divide(img, bg, scale=255)
            img = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]
            cv2.imshow("After", img)
            cv2.imwrite(os.path.join(path,'After.jpg'), img)
            cv2.waitKey(0)
    cv2.destroyAllWindows()
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")


    # print(rect)
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


#################################################################
def correct_orientation(img):


    h, w, c = img.shape
    if (w < h):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w, c = img.shape

    summed = np.sum(255 - img, axis=0)
    img = cv2.resize(img, [480, 280])
    if (np.sum(summed[0:240]) > np.sum(summed[w - 240:w - 0])):
        img = cv2.rotate(img, cv2.ROTATE_180)
    img = cv2.resize(img, [480, 280])
    filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, filter)
    cv2.imwrite('0' + ".jpg", img)
    return img

