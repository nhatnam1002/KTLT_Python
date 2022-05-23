import cv2
import numpy as np
import os
def PreProcessing(img):
    '''
    Đưa hình ảnh về dạng top-down view(từ trên nhìn xuống)
    lưu 2 hình ảnh đã xử lý vào  thư mục image_result
    :param img: hình ảnh chuẩn(1920x2560)
    :return: None
    '''
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

    #img.shape=(W,H)
    w = img.shape[0]
    h = img.shape[1]
    min_area = (w * h) / 10 # tìm area có giá trị =1/10  ảnh

    mask_contours = np.zeros(img.shape) # tạo ma trận 0 với shape=img.shape
    cv2.drawContours(mask_contours, contours, -1, (0, 255, 0), 3) # vẽ contours vào ma trận trên
    mask_contours=cv2.resize(mask_contours,(1280,960)) # resize để không bị tràn hình khi show
    cv2.imshow("Contours", mask_contours) #show hình ảnh
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for cnt in contours:
        contours_approx = cv2.approxPolyDP(cnt,0.01 * cv2.arcLength(cnt, True), True) # bước đơn giản hoá contours
        if len(contours_approx) == 4 and cv2.contourArea(cnt) > min_area: # lọc contours điều kiện là 1 hình chữ nhật và có diện tích lớn hơn 1/10 area

            #Bước vẽ contours sau khi đã qua bước lọc ở trên
            mask_contourss = np.zeros(img.shape) # tạo ma trận 0 với shape=img.shape
            cv2.drawContours(mask_contourss, cnt, -1, (0, 255, 0), 3) # vẽ contours vào ma trận trên
            mask_contourss = cv2.resize(mask_contourss, (1280, 960))  # resize để không bị tràn hình khi show
            cv2.imshow("Contours_Card", mask_contourss) #show hình ảnh
            cv2.waitKey(0)
            cv2.destroyAllWindows()


            extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
            extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
            extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
            extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
            # xác định toạ độ 4 điểm đỉnh nhưng không xác định thứ tự.

            tmp_img = img.copy()
            pts = np.array([extLeft, extRight, extTop, extBot]) # pts chứa toạ độ 4 điểm
            img = four_point_transform(tmp_img, pts)
            # thực hiện phép xoay ma trận để đưa về dạng thẻ sinh viên top-down view
            cv2.imshow("image", img)
            cv2.waitKey(0)

            img = correct_orientation(img)
            # xoay ảnh (90,180 độ) để đưa về dạng chuẩn của thẻ( đúng vị trí 4 đỉnh)
            cv2.imshow("rotated_image", img)
            cv2.waitKey(0)
    path = r'C:\Users\nhatn\PycharmProjects\KTLTPython\image_result' # đường dẫn thư mục lưu hình ảnh đã xử lý

    cv2.imwrite(os.path.join(path, 'Original.jpg'), img)  # lưu lại hình ảnh thẻ trước khi xử lý ocr(top-down view)

    img = ocr_image(img)  # lọc ảnh, tạo ảnh phù hợp để ocr đọc
    cv2.imshow("Ocr image", img)
    cv2.imwrite(os.path.join(path, 'After.jpg'), img)  # lưu lại hình ảnh sau khi xử lý ocr
    cv2.waitKey(0)


    cv2.destroyAllWindows()
def order_points(pts):
    """
        Từ toạ đổ 4 đỉnh ở pts tính toán, sắp xếp thứ tự.
        trả về kết quả là một ma trận chứa 4 toạ độ đỉnh có thứ tự
        0: top left, 1: top right, 2: bottomright 4: bottom left.
        Parameters:
            pts (numpy.ndarray): 4 đỉnh ở mảng pts
        Returns:
             rect (numpy.ndarray): ma trận (2,2)
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1) # tính sum cho từng toạ độ

    rect[0] = pts[np.argmin(s)]# Mặc nhiên cho điểm top-left có sum lớn nhất
    rect[2] = pts[np.argmax(s)]# Mặc nhiên cho điểm top-left có sum nhỏ nhất

    # rect[0] và rect[2] là 2 điểm đầu mút của vector có chiều dài nhất trong 4 điểm
    # tức là 2 đỉnh tạo thành đường chéo

    diff = np.diff(pts, axis=1) #tìm diff

    rect[1] = pts[np.argmin(diff)] # top-right có diff nhỏ nhất
    rect[3] = pts[np.argmax(diff)] #bottom-left có diff lớn nhất

    return rect# return giá trị trên


def four_point_transform(image, pts):
    """
        Từ toạ đổ 4 đỉnh ở pts tính toán, sắp xếp thứ tự.
        Tính toán chiều dài chiều rộng. Từ đó tìm ra được ma trận chuyển đổi.
        Sử dụng ma trận chuyển đổi tìm được để thay đổi góc nhìn của image.
        Parameters:
            image : numpy.ndarray
            pts   : numpy.ndarray
        Returns:
            image : numpy.ndarray

    """
    rect = order_points(pts)
    # gọi hàm order_points(để sắp xếp điểm)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    #tìm chiều rộng lớn nhất

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    #tìm chiều dài lớn nhất
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    #từ chiều dài chiều rộng tạo ra ma trận mẫu thể hiện toạ độ 4 góc của hình
    #chữ nhật (0,0) (Width,0) (Width, Height) ,(0,heigh) ở đây chừa 1 pixel cho đường kẻ

    M = cv2.getPerspectiveTransform(rect, dst) #M là ma trận để xoay matrix
    img = cv2.warpPerspective(image, M, (maxWidth, maxHeight)) # hàm xoay ma trận
    return img


def correct_orientation(img):
    """
        Nếu hình ảnh đang bị dọc (rộng <cao) thì xoay 90 độ
        Sau đó được đưa về chuẩn (480x280), chia đôi hình ảnh, tìm sum ở 2 mảng.
        Vì đặc trưng của ảnh thẻ nên sum của bên phải lớn hơn. Nếu không đúng thì xoay 180 độ.
        Parameters:
            img : numpy.ndarray
        Returns:
            img : numpy.ndarray

    """
    w = img.shape[1]
    h = img.shape[0]
    if w < h:
        # so sánh width và height W<H xoay 90 độ
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        w = img.shape[1]
    img = cv2.resize(img, (480, 280))
    summed = np.sum(255 - img, axis=0)
    if np.sum(summed[0:240]) < np.sum(summed[w - 240:w ]):
        img = cv2.rotate(img, cv2.ROTATE_180)

    return img


def ocr_image(img):
    """
        Đưa hình ảnh về dạng nhị phân trắng đen và khử noise.
        Parameters:
            img : numpy.ndarray.
        Returns:
            img : numpy.ndarray.
     """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kennel = cv2.getStructuringElement(cv2.MORPH_RECT,(8,8)) # tạo kennel
    gradient = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kennel) # tạo bộ khử noise
    gray = cv2.divide(img, gradient, scale=255) # tạo hình ảnh gray scale đã khử noise
    img = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)[1] #đưa về ảnh nhị phân trắng đen

    return img
