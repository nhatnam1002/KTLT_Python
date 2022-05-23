import cv2
import pytesseract
from src.PreProcessing import PreProcessing

img_path=r"demo.jpg"
img = cv2.imread(img_path,1)
# Đọc địa chỉ và hình ảnh. Kích thước chuẩn của hình ảnh input là 1920x2560.

PreProcessing(img)  # module PreProcessing dùng để xử lý hình ảnh.

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # nhập địa chỉ phần mềm ocr

img = cv2.imread(r"image_result\Original.jpg",1)
img_ocr = cv2.imread(r"image_result\After.jpg",0)

# Đọc hình ảnh từ thư mục result chứa hình ảnh sau khi đã được xử lý ở module PreProcessing ở trên
# img chứa hình ảnh không được xử lý ocr
# img_ocr chứa hình ảnh đã xử lý ocr


img_ava = img.copy()

name = cv2.rectangle(img, (160, 126), (475, 168), color=(255, 0, 0), thickness=1)
date = cv2.rectangle(img, (160, 169), (475, 195), color=(255, 0, 0), thickness=1)
major = cv2.rectangle(img, (160, 190), (475, 230), color=(255, 0, 0), thickness=1)
mssv = cv2.rectangle(img, (0, 245), (150, 280), color=(255, 0, 0), thickness=1)
avatar = cv2.rectangle(img, (0, 100), (120, 280), color=(255, 0, 0), thickness=1)

cv2.imshow("frame",img)
cv2.waitKey()
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