import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchsummary import summary
from loss import *
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 50
NUM_WORKERS = 8
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 480
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMAGE_DIR = "/content/drive/MyDrive/IDSegmentation/dataset/data"
TRAIN_MASK_DIR = "/content/drive/MyDrive/IDSegmentation/dataset/data_segment"
VAL_IMAGE_DIR = "/content/drive/MyDrive/IDSegmentation/dataset/data"
VAL_MASK_DIR = "/content/drive/MyDrive/IDSegmentation/dataset/data_segment"
INFO_PATH = "class_dict.csv"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    """
      Hàm khai báo model train cho từng epoch
      Parameter:
        loader(Data.loader)
        model(UNET)
        optimizer,
        loss_fn='FocalLoss',
        scaler
      """
    loop = tqdm(loader)
    for batch_idx, (data,target) in enumerate(loop):
        data = data.to(device=device)
        target = target.float().to(device=device)
        #forward
        target = target.permute(0,3,1,2)

    #forward
        with torch.cuda.amp.autocast():
            predictions = model(data)

            loss = loss_fn(predictions,target)

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    """
    Hàm train
    """
    model = UNET(in_channels=3,out_channels=2,features=[16,32,64,128]).to(device)
    loss_fn = FocalLoss(gamma=3,logits=True)
    optimizer = optim.Adam(model.parameters(),lr=lr)
    
    train_transform = A.Compose(
        [
            ToTensorV2()
        ]
    )
    val_transform = A.Compose(
        [
            # A.Resize(640,480),
            ToTensorV2()
        ]
    )
    train_loader, val_loader = get_loaders(
        TRAIN_IMAGE_DIR, 
        TRAIN_MASK_DIR,
        VAL_IMAGE_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        info_path = INFO_PATH,
        train_transform=train_transform,
        val_transform=val_transform)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        checkpoint = {
            "state_dict":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            }
        save_checkpoints(checkpoint,file_name="checkpoint_FCloss_g5.pth.tar")
        check_accuracy(val_loader, model, epoch=epoch, batch_size=BATCH_SIZE, device=device)
        val_loss(val_loader, model, loss_fn=loss_fn, epoch=epoch, device = device)
        save_predictions_as_imgs(val_loader,model,path="predictions",device=device)
