import torch
import torchvision.transforms.functional as TF
import torch.nn as nn
import numpy as np
import cv2
import pandas as pd
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
    if (w < h):
        # so sánh width và height W<H xoay 90 độ
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        w = img.shape[1]
    img = cv2.resize(img, (480, 280))
    summed = np.sum(255 - img, axis=0)
    if np.sum(summed[0:240]) > np.sum(summed[w - 240:w ]):
        img = cv2.rotate(img, cv2.ROTATE_180)

    return img
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
    # từ toạ đổ 4 điểm(nằm ở 4 điểm trung bình) tính toán toạ độ cho hình rectangle tìm được
    # trả về kết quả là một ma trận chứa 4 toạ độ
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[0]
    rect[1] = pts[2]
    rect[2] = pts[1]
    rect[3] = pts[3]
    return rect

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
        [0, maxHeight - 1]], dtype="float32") #từ chiều dài chiều rộng tạo ra ma trận mẫu thể hiện toạ độ 4 góc của hình
    #chữ nhật (0,0) (Width,0) (Width, Height) ,(0,heigh) ở đây chừa 1 pixel cho đường kẻ

    M = cv2.getPerspectiveTransform(rect, dst) #M là ma trận để xoay matrix
    img = cv2.warpPerspective(image, M, (maxWidth, maxHeight)) # hàm xoay ma trận
    return img


class DoubleConv(nn.Module):
    """
     Khai báo class mạng tích chập Convualtion
    """
    def __init__(self, in_channels, out_channels):
        """
           Hàm constructor cho DoubleConv
           Parameters:
              in_channels (int): số kênh đầu vào.
              out_channels (int): số kênh đầu ra(số labels).
        """
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
          Hàm tính toán lấy output từ input
          Parameters:
             x: input đầu vào từ tensordata.
        """
        return self.conv(x)


class UNET(nn.Module):
    """
       Khai báo class model UNET
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        """
                   Hàm constructor cho model UNET
                   Parameters:
                      in_channels (int): số kênh đầu vào.
                      out_channels (int): số kênh đầu ra(số labels).
                      features(numpy.array) : chỉ các trọng số của mô hình, càng nặng thì chạy càng lâu và chính xác hơn
         """
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # down part of unet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # up part of unet
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
                Hàm tính toán lấy output từ input
                Parameters:
                   x: input đầu vào từ tensordata.
        """
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        # reversed list
        skip_connections = skip_connections[::-1]
        for index in range(0, len(self.ups), 2):
            x = self.ups[index](x)
            skip_connection = skip_connections[index // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[index + 1](concat_skip)

        return self.final_conv(x)


def get_labels_info(info_path):
    """
        Hàm đọc labels và giá trị rgb từ file csv
        Parameters:
            info_path: string
        Returns:
             class_names, labels_values: np.array, np.array
    """
    info = pd.read_csv(info_path)
    class_names = np.array(info["name"])
    labels_values = np.array(info[["r", "g", "b"]])
    return class_names, labels_values


def one_hot_reverse(preds, info_path="class_dict.csv"):
    """
         Từ output là vector onehot đảo thành màu bằng cách đọc file class_dict.csv
         từ giá trị rgb để đảo(reverse) thành màu.
        Parameters:
            info_path: string
        Returns:
             class_names, labels_values: np.array, np.array
    """
    class_names, labels_values = get_labels_info(info_path)
    preds_np = np.array(preds.cpu(), dtype=np.float32)
    img = np.argmax(preds_np, axis=1)
    img_color = labels_values[img.astype(int)]
    return img_color