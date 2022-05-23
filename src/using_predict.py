import pytesseract
from src.Prediction_module import four_point_transform
from src.Prediction_module import UNET
from src.Prediction_module import one_hot_reverse
from src.Prediction_module import correct_orientation
from src.PreProcessing import ocr_image
import os
import torch
from PIL import Image
import numpy as np
import cv2
import pandas as pd
device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNET(in_channels=3,out_channels=2,features=[16,32,64,128]) #load model

img_path=r"C:\Users\nhatn\PycharmProjects\KTLTPython\demo.jpg" #load image
model.load_state_dict(torch.load(r"C:\Users\nhatn\PycharmProjects\KTLTPython\checkpoint_FCloss_g5.pth.tar",map_location=torch.device('cpu'))["state_dict"])
#load weight để predict
img = np.array(Image.open(img_path),dtype=np.float32)
img = cv2.resize(np.float32(img), (1280,960))
#lưu hình ảnh và resize

img = np.expand_dims(img,axis=0)# mở rộng shape của hình ảnh
img = torch.Tensor(img).to(device) # đổi sang dạnhgj tensordata cho hình ảnh

img = img.permute(0,3,1,2)#hoán vị hình ảnh (0,3,1,2)


with torch.no_grad(): # tạm thời set requires_grad flags thành false.
  preds = torch.sigmoid(model(img)) #prediction với dạng vector onehot
  preds = one_hot_reverse(preds,info_path=r"C:\Users\nhatn\PycharmProjects\KTLTPython\class_dict.csv")#đảo onehot thành màu
  for i in range(len(preds)): #show prediction ra màn hình
    cv2.imshow("Prediction",preds[0].astype(np.uint8))
    cv2.waitKey(0)
#Xử lý ảnh
mask= preds[0].astype(np.uint8) #gán mask là prediction
mask = cv2.resize(mask, (1280,960))# resize

img = cv2.imread(img_path)#đọc ảnh gốc
img = cv2.resize(img, (1280,960))#resize ảnh gốc

gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)#đưa về gray scale image
ret, thresh = cv2.threshold(gray, 127, 255, 0, cv2.THRESH_TOZERO_INV)#set thresh

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)#tìm contours

mask_contours = np.zeros(mask.shape)#tạo ma trận zeros
cv2.drawContours(mask_contours, contours, -1, (0,255,0), 3)#vẽ contours
cv2.imshow("Contours",mask_contours) #show contours
cv2.waitKey(0)

w = mask.shape[0]
h = mask.shape[1]
min_area = (w * h) / 10 # tìm area có giá trị =1/10  ảnh


for cnt in contours:
    contours_approx = cv2.approxPolyDP(cnt,0.01 * cv2.arcLength(cnt, True), True)# bước đơn giản hoá contours
    if len(contours_approx) == 4 and cv2.contourArea(cnt) > min_area: # lọc contours điều kiện là 1 hình chữ nhật và có diện tích lớn hơn 1/10 area
        # Bước vẽ contours sau khi đã qua bước lọc ở trên
        mask_contourss = np.zeros(img.shape)  # tạo ma trận 0 với shape=img.shape
        cv2.drawContours(mask_contourss, cnt, -1, (0, 255, 0), 3)  # vẽ contours vào ma trận trên
        mask_contourss = cv2.resize(mask_contourss, (1280, 960))  # resize để không bị tràn hình khi show
        cv2.imshow("Contours_Card", mask_contourss)  # show hình ảnh
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
        extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
        extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
        extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
        # xác định toạ độ 4 điểm đỉnh nhưng không xác định thứ tự.

        tmp_mask = mask.copy()
        tmp_img = img.copy()
        pts = np.array([extLeft, extRight, extTop, extBot])# np.array chứa toạ độ 4 điểm
        # sử dung ma trận để xác định ảnh dạng thẻ sinh viên


        mask = four_point_transform(tmp_mask, pts) # thực hiện phép xoay ma trận để đưa về dạng thẻ sinh viên top-down view
        mask = correct_orientation(mask)   # xoay ảnh (90,180 độ) để đưa về dạng chuẩn của thẻ( đúng vị trí 4 đỉnh)

        img = four_point_transform(tmp_img, pts) # thực hiện phép xoay ma trận để đưa về dạng thẻ sinh viên top-down view
        cv2.imshow("image", img)
        cv2.waitKey(0)
        img = correct_orientation(img) # xoay ảnh (90,180 độ) để đưa về dạng chuẩn của thẻ( đúng vị trí 4 đỉnh)
        cv2.imshow("rotated_imagee", img)
        cv2.waitKey(0)



        path= r'C:\Users\nhatn\PycharmProjects\KTLTPython\image_result'

        cv2.imwrite(os.path.join(path, 'Originall.jpg'), img)  # lưu lại hình ảnh thẻ trước khi xử lý ocr(top-down view)

        img = ocr_image(img)  # lọc ảnh, tạo ảnh phù hợp để ocr đọc
        cv2.imshow("Ocr image", img)
        cv2.imwrite(os.path.join(path, 'Afterr.jpg'), img)  # lưu lại hình ảnh sau khi xử lý ocr
        cv2.waitKey(0)

        cv2.destroyAllWindows()
#chay ocr
img = cv2.imread(r"C:\Users\nhatn\PycharmProjects\KTLTPython\image_result\Originall.jpg",1)
img_ocr = cv2.imread(r"C:\Users\nhatn\PycharmProjects\KTLTPython\image_result\Afterr.jpg",0)
img_ava = img.copy()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'# nhập địa chỉ phần mềm ocr

name = cv2.rectangle(img, (160, 126), (475, 168), color=(255, 0, 0), thickness=1)
date = cv2.rectangle(img, (160, 169), (475, 195), color=(255, 0, 0), thickness=1)
major = cv2.rectangle(img, (160, 190), (475, 230), color=(255, 0, 0), thickness=1)
mssv = cv2.rectangle(img, (0, 245), (150, 280), color=(255, 0, 0), thickness=1)
avatar = cv2.rectangle(img, (0, 100), (120, 280), color=(255, 0, 0), thickness=1)


cv2.imshow("frame",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Sử dụng cv2.rectangle để khoanh vùng hình ảnh mong muốn trên ảnh.
# Bước này giúp người dùng có cái nhìn trực quan về các vùng được chia, không ảnh hưởng đến các bước xử lý sau.



name_ocr = img_ocr [126:168,160:475]
date_ocr = img_ocr [166:198, 190:475]
major_ocr = img_ocr [190:230, 160:475]
mssv_ocr = img_ocr [240:280, 0:150]
avatar = img_ava[100:280, 0:120]
# dùng kĩ thuật slicing để tách vùng văn bản, hình ảnh.
# img_ocr được chia thành 5 hình ảnh nhỏ tương ứng với tên, niên khoá, ngành, mã số sinh viên, avatar



custom_config = r'--psm 6 --oem 3 '  # tạo config cho bộ ocr.
name = pytesseract.image_to_string(name_ocr, lang='vie', config=custom_config) # chạy OCR và lưu giá trị do OCR đọc được
print(name)
custom_config = r'--psm 6 --oem 3 ' # tạo config cho bộ ocr.
date = pytesseract.image_to_string(date_ocr, lang='vie', config=custom_config) # chạy OCR và lưu giá trị do OCR đọc được
print(date)
custom_config = r'--psm  7  --oem 3'   # tạo config cho bộ ocr.
major = pytesseract.image_to_string(major_ocr, lang='vie', config=custom_config) # chạy OCR và lưu giá trị do OCR đọc được
print(major)
custom_config = r'--psm  6 --oem 3 -c tessedit_char_whitelist=0123456789'
# tạo config cho bộ ocr
# tạo bộ whitelist chỉ cho phép đọc số
mssv = pytesseract.image_to_string(mssv_ocr, lang='vie', config=custom_config) # chạy OCR và lưu giá trị do OCR đọc được
print(mssv)

cv2.waitKey()
