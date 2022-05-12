import cv2
import pytesseract
from src.PreProcessing import PreProcessing
import pandas as pd
import numpy as np
img = cv2.imread(r'C:\Users\nhatn\PycharmProjects\KTLTPython\image_test\test_(19).jpg ', 1)
# cv2.imshow("fw",img)
# cv2.waitKey()
PreProcessing(img)
img = cv2.imread(r"C:\Users\nhatn\PycharmProjects\KTLTPython\image_result\After.jpg",1)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
img = cv2.imread(r"C:\Users\nhatn\PycharmProjects\KTLTPython\image_result\Original.jpg",1)
img_ocr = cv2.imread(r"C:\Users\nhatn\PycharmProjects\KTLTPython\image_result\After.jpg",0)
img_ava = img.copy()


name = cv2.rectangle(img, (160, 126), (475, 168), color=(255, 0, 0), thickness=1)
date = cv2.rectangle(img, (160, 169), (475, 195), color=(255, 0, 0), thickness=1)
major = cv2.rectangle(img, (160, 190), (475, 230), color=(255, 0, 0), thickness=1)
mssv = cv2.rectangle(img, (0, 245), (150, 280), color=(255, 0, 0), thickness=1)
avatar = cv2.rectangle(img, (0, 100), (120, 280), color=(255, 0, 0), thickness=1)


# cv2.imshow("frame",mssv)
# cv2.waitKey()


name_ocr = img_ocr [126:168 ,160:475]
date_ocr = img_ocr [166:198 , 190:475]
major_ocr = img_ocr [190:230 , 160:475]
mssv_ocr = img_ocr [240:280 , 0:150]
avatar = img_ava[100:280, 0:120]


custom_config = r'--oem 3 --psm 6'
name = pytesseract.image_to_string(name_ocr, lang='vie', config=custom_config)
print(name)
custom_config = r'--psm 6 --oem 3 ' # number only
date = pytesseract.image_to_string(date_ocr, lang='vie', config=custom_config)
print(date)
custom_config = r'--psm 10 --oem 3'  # 12    Sparse text with OSD.
major = pytesseract.image_to_string(major_ocr, lang='vie', config=custom_config)
print(major)
custom_config = r'--psm  6 --oem 3 -c tessedit_char_whitelist=0123456789' # number only
mssv = pytesseract.image_to_string(mssv_ocr, lang='vie', config=custom_config)
print(mssv)

custom_config = r'--oem 3 --psm 6'
name = pytesseract.image_to_string(img, lang='vie', config=custom_config)
print(name)

cv2.waitKey()