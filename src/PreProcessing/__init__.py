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
    #contours = array các vector điểm nối các điểm tạo thành đường cong
    #tập các điểm-liên-tục tạo thành một đường cong (curve) gọi là contours. Việc tìm contours giúp tìm ra các object có trong hình.
    #hàm trên sẽ lưu các đường biên( vector của các điểm) ở array contours, hierachy lưu giá trị đường viền(không sử dụng)

    #img.shape=(W,H,C)
    w = img.shape[0]
    h = img.shape[1]
    min_area = (w * h) / 10 # tìm area có giá trị =1/10  ảnh

    for cnt in contours:
        contours_approx = cv2.approxPolyDP(cnt,0.01 * cv2.arcLength(cnt, True), True) # bước đơn giản hoá contours
        if len(contours_approx) == 4 and cv2.contourArea(cnt) > min_area: # điều kiện là 1 hình chữ nhật và có diện tích lớn hơn 1/10 area


            extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
            extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
            extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
            extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
            # xác định toạ độ 4 điểm.

            tmp_img = img.copy()
            pts = np.array([extLeft, extRight, extTop, extBot]) # np.array chứa toạ độ 4 điểm
            img = four_point_transform(tmp_img, pts) # sử dung ma trận để xác định ảnh dạng thẻ sinh viên
            img = correct_orientation(img)#xoay ảnh


            path=r'C:\Users\nhatn\PycharmProjects\KTLTPython\image_result'


            cv2.imshow("Original", img)
            cv2.imwrite(os.path.join(path , 'Original.jpg'), img) #lưu lại hình ảnh thẻ trước khi xử lý ocr(top down view)
            cv2.waitKey(0)

            img = ocr_image(img) # lọc ảnh, tạo ảnh phù hợp để ocr đọc


            cv2.imshow("After", img)
            cv2.imwrite(os.path.join(path,'After.jpg'), img)#lưu lại hình ảnh sau khi xử lý ocr
            cv2.waitKey(0)


    cv2.destroyAllWindows()
def order_points(pts):
    # từ toạ đổ 4 điểm(nằm ở 4 điểm trung bình) tính toán toạ độ cho hình rectangle tìm được
    # trả về kết quả là một ma trận chứa 4 toạ độ
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # t = 1;
    # for i in (pts):
    #     print("a=", np.float32(i), "rect[0]", rect[0], "rect[2]", rect[2])
    #     if not (np.array_equal(np.float32(i), rect[0]) or np.array_equal(np.float32(i), rect[2])):
    #         print("i=", i)
    #         rect[t] = np.float32(i)
    #         print()
    #         t = t + 2
    # print("rect", rect)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32") #từ chiều dài chiều rộng tạo ra ma trận mẫu thể hiện toạ độ 4 góc của hình
    #chữ nhật (0,0) (Width,0) (Width, Height) ,(0,heigh) ở đây chừa 1 pixel cho đường kẻ

    M = cv2.getPerspectiveTransform(rect, dst) #M là ma trận để xoay matrix
    img = cv2.warpPerspective(image, M, (maxWidth, maxHeight)) # hàm xoay ma trận
    return img


def correct_orientation(img):
    w = img.shape[1]
    h =img.shape[0]
    if (w < h):
        # so sánh width và height W<H xoay 90 độ
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        w = img.shape[1]

    summed = np.sum(255 - img, axis=0)
    img = cv2.resize(img, [480, 280])

    if np.sum(summed[0:240]) > np.sum(summed[w - 240:w - 0]):
        img = cv2.rotate(img, cv2.ROTATE_180)

    return img


def ocr_image(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    bg = cv2.morphologyEx(img, cv2.MORPH_DILATE, se)
    out_gray = cv2.divide(img, bg, scale=255)
    # filter tách lọc hình loại bỏ background và đưa về ảnh nhị phân ( trắng, đen)
    img = cv2.threshold(out_gray,0,255,cv2.THRESH_OTSU)[1]

    return img
