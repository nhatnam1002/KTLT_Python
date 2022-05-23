
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import cv2
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
    labels_values = np.array(info[["r","g","b"]])
    return class_names, labels_values

def convert_data(img, label, info_path):
    """
            Hàm đọc chuyển đổi data (img=img/255, loại bỏ label trùng)
            Parameters:
                img: np.array
                label:np.array
                info_path: string
            Returns:
                 img, semantic_map(label)
     """
    img = img/255.0
    class_names, labels_values = get_labels_info(info_path)
    sematic_maps = []
    for color in labels_values:
        same = np.equal(label, color)
        class_map = np.all(same,axis=-1)
        sematic_maps.append(class_map)
    semantic_map = np.array(np.stack(sematic_maps,axis=-1))
    return img, semantic_map

class RC_dataset(Dataset):
    """
         Khai báo class RC_dataset
    """
    def __init__(self, image_dir, label_dir, info_path, transform = None):
        """
               Hàm constructor
                  Parameters:
                  image_dir(string): địa chỉ hình ảnh,
                  label_dir (string): địa chỉ label,
                  info_path(string): địa chỉ file class_dict,
                  transform(bool)
               """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.info_path = info_path


    def __len__(self):
        """
             Hàm tính số ảnh
             :return: số ảnh (int)
        """
        return len(self.images)
    
    def __getitem__(self,idx):
        """
            Hàm biến đổi dataset, dùng các argument...
            Parameters:
             idx:vị trí
            Return: image(np.array),
                    label(np.array)
        """
        img_path = os.path.join(self.image_dir,self.images[idx])

        #change tail name
        path_name = self.images[idx].replace(".jpg","_converted.png")
        path_name = path_name.replace(".jpeg","_converted.png")

        label_path = os.path.join(self.label_dir,path_name )
        img = np.array(Image.open(img_path),dtype=np.float32)
        # img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        label = np.array(Image.open(label_path),dtype=np.float32)
        img = cv2.resize(np.float32(img), (1280,960))
        label = cv2.resize(np.float32(label), (1280,960))
        img, label = convert_data(img, label, self.info_path)
        if self.transform is not None:
            augmentation = self.transform(image=img,mask=label)
            img = augmentation["image"]
            label = augmentation["mask"]

        return img, label