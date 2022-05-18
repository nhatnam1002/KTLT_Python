import torch
from torch.nn.modules import batchnorm
from torch.nn.modules.activation import ReLU
import torchvision.transforms.functional as TF
import torch.nn as nn
import numpy as np
import cv2
from torchsummary import summary
import pandas as pd
def order_points(pts):
    # từ toạ đổ 4 điểm(nằm ở 4 điểm trung bình) tính toán toạ độ cho hình rectangle tìm được
    # trả về kết quả là một ma trận chứa 4 toạ độ
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    t = 1
    for i in (pts):
        print("a=", np.float32(i), "rect[0]", rect[0], "rect[2]", rect[2])
        if not (np.array_equal(np.float32(i), rect[0]) or np.array_equal(np.float32(i), rect[2])):
            print("i=", i)
            rect[t] = np.float32(i)
            print()
            t = t + 2
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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
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
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
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
    info = pd.read_csv(info_path)
    # info has format: [['obstacles' 59 193 246]...]
    # info = info.to_numpy()
    class_names = np.array(info["name"])
    labels_values = np.array(info[["r", "g", "b"]])
    return class_names, labels_values


def one_hot_reverse(preds, info_path="class_dict.csv"):
    class_names, labels_values = get_labels_info(info_path)
    preds_np = np.array(preds.cpu(), dtype=np.float32)
    img = np.argmax(preds_np, axis=1)
    img_color = labels_values[img.astype(int)]
    # img_color = torch.Tensor(img_color).permute(0,3,1,2)
    return img_color