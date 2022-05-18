import cv2
import pytesseract
from src.Prediction_module import four_point_transform
from src.Prediction_module import UNET
from src.Prediction_module import one_hot_reverse
from src.PreProcessing import ocr_image
from src.PreProcessing import correct_orientation
import numpy as np
import os
import torch
from PIL import Image
import numpy as np
import cv2
import pandas as pd
#doc mask tu model.predict
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNET(in_channels=3,out_channels=2,features=[16,32,64,128])
model.load_state_dict(torch.load(r"C:\Users\nhatn\PycharmProjects\KTLTPython\checkpoint_FCloss_g5.pth.tar",map_location=torch.device('cpu'))["state_dict"])
img = np.array(Image.open(r"C:\Users\nhatn\PycharmProjects\KTLTPython\8"
                          r"..jpg"),dtype=np.float32)
img = cv2.resize(img, (1280,960))
#img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
img = np.expand_dims(img,axis=0)
img = torch.Tensor(img).to(device)

img = img.permute(0,3,1,2)


with torch.no_grad():
  preds = torch.sigmoid(model(img))
  preds = one_hot_reverse(preds,info_path=r"C:\Users\nhatn\PycharmProjects\KTLTPython\class_dict.csv")
  for i in range(len(preds)):
    #BGRpreds = cv2.cvtColor(preds[i],cv2.COLOR_RGB2BGR)
    #cv2.imwrite("test.png",preds)
    print(preds[0].shape)

    cv2.imshow("Prediction",preds[0].astype(np.uint8))
    cv2.waitKey(0)
#xu ly anh
mask= preds[0].astype(np.uint8)
img = cv2.imread(r"C:\Users\nhatn\PycharmProjects\KTLTPython\8.jpg")
img = cv2.resize(img, (1280,960))
gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 127, 255, 0, cv2.THRESH_TOZERO_INV)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

mask_contours = np.zeros(mask.shape)
cv2.drawContours(mask_contours, contours, -1, (0,255,0), 3)
cv2.imshow("sdw",mask_contours)

w = mask.shape[0]
h = mask.shape[1]
min_area = (w * h) / 6 # tìm area có giá trị =1/10  ảnh


for cnt in contours:
    contours_approx = cv2.approxPolyDP(cnt,0.01 * cv2.arcLength(cnt, True), True)# bước đơn giản hoá contours
    if len(contours_approx) >= 4 and cv2.contourArea(cnt) > min_area: # điều kiện là 1 hình chữ nhật và có diện tích lớn hơn 1/10 area
        extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
        extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
        extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
        extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
        # xác định toạ độ 4 điểm.

        tmp_mask = mask.copy()
        tmp_img = img.copy()
        pts = np.array([extLeft, extRight, extTop, extBot])
        print(pts)# np.array chứa toạ độ 4 điểm
        # sử dung ma trận để xác định ảnh dạng thẻ sinh viên
        mask = four_point_transform(tmp_mask, pts)
        mask = correct_orientation(mask)  # xoay ảnh
        cv2.imshow("image",img)
        cv2.imshow("mask",mask)
        img = four_point_transform(tmp_img, pts)
        img = correct_orientation(img)

        path= r'/image_result'


        cv2.imshow("Original", img)
        cv2.imwrite(os.path.join(path , 'Original.jpg'), img) #lưu lại hình ảnh thẻ trước khi xử lý ocr(top down view)
        cv2.waitKey(0)
        img = ocr_image(img) # lọc ảnh, tạo ảnh phù hợp để ocr đọc
        #
        cv2.imshow("After", img)
        cv2.imwrite(os.path.join(path,'After.jpg'), img)#lưu lại hình ảnh sau khi xử lý ocr
        cv2.waitKey(0)
#chay ocr
img = cv2.imread(r"C:\Users\nhatn\PycharmProjects\KTLTPython\image_result\Original.jpg",1)
img_ocr = cv2.imread(r"C:\Users\nhatn\PycharmProjects\KTLTPython\image_result\After.jpg",0)
img_ava = img.copy()


name = cv2.rectangle(img, (160, 126), (475, 168), color=(255, 0, 0), thickness=1)
date = cv2.rectangle(img, (160, 169), (475, 195), color=(255, 0, 0), thickness=1)
major = cv2.rectangle(img, (160, 190), (475, 230), color=(255, 0, 0), thickness=1)
mssv = cv2.rectangle(img, (0, 245), (150, 280), color=(255, 0, 0), thickness=1)
avatar = cv2.rectangle(img, (0, 100), (120, 280), color=(255, 0, 0), thickness=1)


cv2.imshow("frame",img)
cv2.waitKey()


name_ocr = img_ocr [126:168,160:475]
date_ocr = img_ocr [166:198, 190:475]
major_ocr = img_ocr [190:230, 160:475]
mssv_ocr = img_ocr [240:280, 0:150]
avatar = img_ava[100:280, 0:120]
# dùng slicing để tách vùng văn bản, hình ảnh.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

custom_config = r'--psm 6 --oem 3 '
name = pytesseract.image_to_string(name_ocr, lang='vie', config=custom_config)
print(name)
custom_config = r'--psm 6 --oem 3 '
date = pytesseract.image_to_string(date_ocr, lang='vie', config=custom_config)
print(date)
custom_config = r'--psm  7  --oem 3'  # 12    Sparse text with OSD.
major = pytesseract.image_to_string(major_ocr, lang='vie', config=custom_config)
print(major)
custom_config = r'--psm  6 --oem 3 -c tessedit_char_whitelist=0123456789' #Chỉ số
mssv = pytesseract.image_to_string(mssv_ocr, lang='vie', config=custom_config)
print(mssv)


cv2.waitKey()
#
#
#
