
import torch
from dataset import *
from torch.utils.data import DataLoader
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"


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


def save_checkpoints(state, file_name="checkpoint.pth.tar"):
    """
              Hàm save checkpoints
              Parameters:
                  state: torch.sensordata
                  file_name:string
       """
    print("=========> saving checkpoint")
    torch.save(state,file_name)

def load_checkpoints(checkpoint, model):
    """
              Hàm load checkpoints
              Parameters:
                  model: load_state_dict
                  checkpoint:torch.sensordata
       """
    print("==========> loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    return model

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    info_path,
    num_workers=4,
    pin_memory=True,
):
    """
         Hàm đặt địa chỉ, tham số cho việc train.
         Parameters:
             train_dir,
            train_maskdir,
            val_dir,
            val_maskdir,
            batch_size,
            train_transform,
            val_transform,
            info_path,
            num_workers
            pin_memory,
         Returns:
             train_loader:DataLoader,
             val_loader:DataLoader
          """
    train_ds = RC_dataset(
        image_dir=train_dir,
        label_dir = train_maskdir,
        info_path= info_path,
        transform=train_transform
    )
    val_ds = RC_dataset(
        image_dir=val_dir,
        label_dir= val_maskdir,
        info_path = info_path,
        transform=val_transform
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader,val_loader

def check_accuracy(loader, model, epoch, batch_size = 8, device="cuda"):
    """
                Hàm tính độ chính xác
                Parameters:
                    loader(DataLoader)
                    model(UNET),
                    epoch(int),
                    batch_size(int),
                    device="cuda"
         """
    num_correct = 0
    num_pixels = 0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            y = y.permute(0,3,1,2)
            preds = torch.sigmoid(model(x))
            preds = (preds>0.5).float()
            num_correct += (preds==y).sum()
            num_pixels += torch.numel(preds)
    print(f"EPOCH: {epoch} Got {num_correct}/{num_pixels*batch_size} ----> accuracy = {num_correct/num_pixels*100:.2f}")
    model.train()

def one_hot_reverse(preds,info_path="class_dict.csv"):
    """
           Từ output là vector onehot đảo thành màu bằng cách đọc file class_dict.csv
           từ giá trị rgb để đảo(reverse) thành màu.
          Parameters:
              info_path: string
          Returns:
               class_names, labels_values: np.array, np.array
      """
    class_names, labels_values = get_labels_info(info_path)
    preds_np = np.array(preds.cpu(),dtype=np.float32)
    img = np.argmax(preds_np,axis=1)
    img_color = labels_values[img.astype(int)]
    #img_color = torch.Tensor(img_color).permute(0,3,1,2)
    return img_color

def save_predictions_as_imgs(loader, model, path = "predictions",device = "cuda",info_path="class_dict.csv"):
    """
          Chuyển prediction thành hình ảnh
          Parameters:
              loader(DataLoader),
              model(UNET),
              path(string),
              device="cuda"
      """
    model.eval()
    
    for idx, (x,y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            #preds = (preds >0.5).float()
        preds = one_hot_reverse(preds)
        for i in range(len(preds)):
            #BGRpreds = cv2.cvtColor(preds[i],cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{path}/model_predict/pred_{idx}_{i}.png",preds[i])
        # torchvision.utils.save_image(y.unsqueeze(1),f"{path}/label/label_{idx}.png")
    model.train()

def val_loss(loader, model, loss_fn, epoch, device="cuda"):
    """
            Tính val_loss
            Parameters:
                loader(DataLoader),
                model(UNET),
                loss_fn='FocalLoss',
                epoch(int)
                device="cuda"
        """
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            y = y.float().to(device = device)
            y = y.permute(0,3,1,2)
            preds = model(x)
            loss = loss_fn(y, preds)
            loss_total+= loss
    print(f"total loss in val dataset at epoch {epoch}: {loss_total}")
    return loss_total


