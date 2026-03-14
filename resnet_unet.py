import os
import cv2
import time
import csv
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from skimage import color
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as calc_mse

IMAGE_SIZE = 96
NUM_EPOCHS = 50
BATCH_SIZE = 32
L_NORM = 50.0
AB_NORM = 110.0

IMAGENET_MEAN = 0.449
IMAGENET_STD = 0.226

CHECKPOINT_DIR = "checkpoints/checkpoints_resnet_unet"
RESULTS_DIR = "results/results_resnet_unet"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = Path("archive")
train_dir = base_dir / "train_images"
test_dir = base_dir / "test_images"
unlabeled_dir = base_dir / "unlabeled_images"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "val_samples"), exist_ok=True)

def augment_image(image):
    if random.random() > 0.5:
        image = TF.hflip(image)
    crop_size = int(IMAGE_SIZE * 0.85)
    top, left, h, w = T.RandomCrop.get_params(image, (crop_size, crop_size))
    image = TF.crop(image, top, left, h, w)
    image = TF.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    return image

class ColorizationDataset(Dataset):

    def __init__(self, image_paths, is_train=False):
        self.image_paths = image_paths
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = str(self.image_paths[idx])
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        if self.is_train:
            img_pil = Image.fromarray(img)
            img_pil = augment_image(img_pil)
            img = np.array(img_pil)
        img = img.astype(np.float32) / 255.0
        lab = color.rgb2lab(img).astype(np.float32)
        L  = lab[:, :, 0] / L_NORM - 1.0
        AB = lab[:, :, 1:] / AB_NORM
        L_tensor  = torch.from_numpy(L).unsqueeze(0)
        AB_tensor = torch.from_numpy(AB).permute(2, 0, 1)
        return L_tensor, AB_tensor

def get_dataloaders():
    train_images = list(train_dir.glob("*.*"))
    test_images = list(test_dir.glob("*.*"))
    unlabeled_imgs = list(unlabeled_dir.glob("*.*"))
    print(f"Labeled training images : {len(train_images)}")
    print(f"Unlabeled images : {len(unlabeled_imgs)}")
    print(f"Test images : {len(test_images)}")
    num_val = max(1, int(0.1 * len(train_images)))
    val_imgs = train_images[:num_val]
    train_imgs = train_images[num_val:] + unlabeled_imgs
    random.shuffle(train_imgs)
    train_loader = DataLoader(ColorizationDataset(train_imgs, is_train=True),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(ColorizationDataset(val_imgs,   is_train=False),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(ColorizationDataset(test_images, is_train=False),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches : {len(val_loader)}")
    print(f"Test batches : {len(test_loader)}")
    return train_loader, val_loader, test_loader

def make_decoder_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return block

class ResNetUNet(nn.Module):

    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  
        self.pool = resnet.maxpool
        self.enc1 = resnet.layer1  
        self.enc2 = resnet.layer2  
        self.enc3 = resnet.layer3  
        self.enc4 = resnet.layer4 
        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = make_decoder_block(256 + 256, 256) 
        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = make_decoder_block(128 + 128, 128)  
        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = make_decoder_block(64 + 64, 64)    
        self.up1  = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = make_decoder_block(64 + 64, 64)  
        self.up0  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = make_decoder_block(32, 32)
        self.output_layer = nn.Conv2d(32, 2, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x3 = x.repeat(1, 3, 1, 1)
        x3 = (x3 + 1.0) / 2.0                      
        x3 = (x3 - IMAGENET_MEAN) / IMAGENET_STD    
        e0 = self.enc0(x3)            
        e1 = self.enc1(self.pool(e0)) 
        e2 = self.enc2(e1)             
        e3 = self.enc3(e2)           
        e4 = self.enc4(e3)          
        d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e0], dim=1))
        d0 = self.dec0(self.up0(d1))
        output = self.tanh(self.output_layer(d0))
        return output

def lab_to_rgb_differentiable(L_tensor, AB_tensor):
    L_raw  = (L_tensor + 1.0) * L_NORM
    AB_raw = AB_tensor * AB_NORM
    a_channel = AB_raw[:, 0:1]
    b_channel = AB_raw[:, 1:2]
    delta = 6.0 / 29.0
    fy = (L_raw + 16.0) / 116.0
    fx = fy + a_channel / 500.0
    fz = fy - b_channel / 200.0
    def f_inv(t):
        return torch.where(t > delta, t ** 3, 3.0 * delta ** 2 * (t - 4.0 / 29.0))
    X = 0.95047 * f_inv(fx)
    Y = 1.00000 * f_inv(fy)
    Z = 1.08883 * f_inv(fz)
    r =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    b =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
    rgb = torch.cat([r, g, b], dim=1).clamp(min=1e-8, max=1.0)
    rgb = (rgb ** (1.0 / 2.2)).clamp(0.0, 1.0)
    return rgb

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:9])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, L, AB_pred, AB_true):
        pred_rgb = lab_to_rgb_differentiable(L, AB_pred)
        with torch.no_grad():
            true_rgb = lab_to_rgb_differentiable(L, AB_true)
        pred_rgb = (pred_rgb - self.mean) / self.std
        true_rgb = (true_rgb - self.mean) / self.std
        pred_features = self.feature_extractor(pred_rgb)
        true_features = self.feature_extractor(true_rgb)
        loss = nn.functional.l1_loss(pred_features, true_features)
        return loss

class CombinedLoss(nn.Module):

    def __init__(self, perceptual_weight=0.05):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual = PerceptualLoss()
        self.perc_weight = perceptual_weight

    def forward(self, L, AB_pred, AB_true):
        l1 = self.l1_loss(AB_pred, AB_true)
        perc = self.perceptual(L, AB_pred, AB_true)
        total = l1 + self.perc_weight * perc
        return total, l1, perc

def tensors_to_rgb(L_batch, AB_batch):
    rgb_list = []
    L_np = L_batch.detach().cpu().numpy()
    AB_np = AB_batch.detach().cpu().numpy()
    for i in range(L_np.shape[0]):
        lab = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        lab[:, :, 0] = (L_np[i, 0] + 1.0) * L_NORM
        lab[:, :, 1:] = AB_np[i].transpose(1, 2, 0) * AB_NORM
        rgb = np.clip(color.lab2rgb(lab), 0, 1)
        rgb_list.append(rgb)
    return rgb_list

def compute_metrics(L, AB_pred, AB_true):
    pred_rgbs = tensors_to_rgb(L, AB_pred)
    true_rgbs = tensors_to_rgb(L, AB_true)
    mse_list = []
    ssim_list = []
    for pred_rgb, true_rgb in zip(pred_rgbs, true_rgbs):
        mse_list.append(calc_mse(true_rgb, pred_rgb))
        ssim_list.append(ssim(true_rgb, pred_rgb, channel_axis=2, data_range=1.0))
    return float(np.mean(mse_list)), float(np.mean(ssim_list))

def save_sample_grid(L, AB_true, AB_pred, save_path, num_images=4):
    num_images = min(num_images, L.shape[0])
    true_rgbs = tensors_to_rgb(L[:num_images], AB_true[:num_images])
    pred_rgbs = tensors_to_rgb(L[:num_images], AB_pred[:num_images])
    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 3, 9))
    for i in range(num_images):
        gray_img = (L[i, 0].cpu().numpy() + 1.0) * L_NORM
        axes[0, i].imshow(gray_img, cmap='gray', vmin=0, vmax=100)
        axes[1, i].imshow(true_rgbs[i])
        axes[2, i].imshow(pred_rgbs[i])
        for row in range(3):
            axes[row, i].axis('off')
    axes[0, 0].set_ylabel("Grayscale Input")
    axes[1, 0].set_ylabel("Ground Truth")
    axes[2, 0].set_ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved sample grid : {save_path}")

def save_checkpoint(epoch, model, optimizer, scheduler, val_loss, is_best):
    state = {
        'epoch' : epoch,
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict(),
        'val_loss' : val_loss
    }
    torch.save(state, os.path.join(CHECKPOINT_DIR, 'last.pth'))
    if is_best:
        torch.save(state, os.path.join(CHECKPOINT_DIR, 'best.pth'))
        print(f"New best model saved! val_loss = {val_loss:.5f}")


def load_checkpoint(model, optimizer, scheduler):
    path = os.path.join(CHECKPOINT_DIR, 'last.pth')
    if not os.path.exists(path):
        print("No checkpoint found. Starting from scratch.")
        return 1, float('inf')

    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['val_loss']
    print(f"Resumed from epoch {checkpoint['epoch']}, val_loss = {best_loss:.5f}")

    return start_epoch, best_loss

def train(resume=False):
    print("=" * 50)
    print("ResNet18 U-Net - Image Colorization")
    print(f"Device : {DEVICE}")
    print(f"Epochs : {NUM_EPOCHS}")
    print(f"Batch : {BATCH_SIZE}")
    print("=" * 50)
    train_loader, val_loader, _ = get_dataloaders()
    model = ResNetUNet().to(DEVICE)
    criterion = CombinedLoss(perceptual_weight=0.05).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}\n")
    encoder_params = (list(model.enc0.parameters()) +
                      list(model.enc1.parameters()) +
                      list(model.enc2.parameters()) +
                      list(model.enc3.parameters()) +
                      list(model.enc4.parameters()))
    decoder_params = [p for p in model.parameters()
                      if not any(p is ep for ep in encoder_params)]
    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': 1e-4}, 
        {'params': decoder_params, 'lr': 1e-3}, 
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    start_epoch   = 1
    best_val_loss = float('inf')
    if resume:
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler)
    log_path = os.path.join(RESULTS_DIR, "training_log.csv")
    log_file = open(log_path, 'a' if resume else 'w', newline='')
    log_csv  = csv.writer(log_file)
    if not resume:
        log_csv.writerow(['epoch', 'train_loss', 'val_loss', 'val_mse', 'val_ssim', 'lr_enc', 'lr_dec'])
    all_train_losses = []
    all_val_losses = []
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        for L, AB_true in tqdm(train_loader, desc=f"Epoch {epoch:03d} [Train]", leave=False):
            L = L.to(DEVICE)
            AB_true = AB_true.to(DEVICE)
            AB_pred = model(L)
            loss, l1, per = criterion(L, AB_pred, AB_true)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        train_loss = total_train_loss / len(train_loader)
        model.eval()
        total_val_loss = 0.0
        total_val_mse  = 0.0
        total_val_ssim = 0.0
        saved_grid = False

        with torch.no_grad():
            for L, AB_true in tqdm(val_loader, desc=f"Epoch {epoch:03d} [Val]  ", leave=False):
                L = L.to(DEVICE)
                AB_true = AB_true.to(DEVICE)
                AB_pred = model(L)
                loss, l1, per = criterion(L, AB_pred, AB_true)
                total_val_loss += loss.item()
                batch_mse, batch_ssim = compute_metrics(L, AB_pred, AB_true)
                total_val_mse += batch_mse
                total_val_ssim += batch_ssim
                if not saved_grid and (epoch % 5 == 0 or epoch == 1):
                    grid_path = os.path.join(RESULTS_DIR, "val_samples", f"epoch_{epoch:03d}.png")
                    save_sample_grid(L, AB_true, AB_pred, grid_path)
                    saved_grid = True
        val_loss = total_val_loss / len(val_loader)
        val_mse = total_val_mse / len(val_loader)
        val_ssim = total_val_ssim / len(val_loader)
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < prev_lr:
            print(f"Learning rate reduced")
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        save_checkpoint(epoch, model, optimizer, scheduler, val_loss, is_best)
        lr_enc = optimizer.param_groups[0]['lr']
        lr_dec = optimizer.param_groups[1]['lr']
        log_csv.writerow([epoch, train_loss, val_loss, val_mse, val_ssim, lr_enc, lr_dec])
        log_file.flush()
        all_train_losses.append(train_loss)
        all_val_losses.append(val_loss)
        elapsed = time.time() - epoch_start
        marker = "*" if is_best else " "
        print(f"Epoch {epoch:03d} {marker} | "
              f"Train: {train_loss:.5f} | "
              f"Val: {val_loss:.5f} | "
              f"MSE: {val_mse:.5f} | "
              f"SSIM: {val_ssim:.4f} | "
              f"LR: {lr_enc:.2e}/{lr_dec:.2e} | "
              f"{elapsed:.0f}s")

    log_file.close()

    epochs = list(range(1, len(all_train_losses) + 1))
    plt.figure(figsize=(9, 4))
    plt.plot(epochs, all_train_losses, label='Train Loss', color='steelblue')
    plt.plot(epochs, all_val_losses,   label='Val Loss',   color='tomato')
    plt.xlabel('Epoch')
    plt.ylabel('Combined Loss')
    plt.title('ResNet18 U-Net - Training and Validation Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"), dpi=120)
    plt.close()
    print(f"Loss curve saved : {RESULTS_DIR}/loss_curve.png")

    print(f"\nTraining complete!")
    print(f"Best val loss : {best_val_loss:.5f}")
    print(f"Checkpoints : {CHECKPOINT_DIR}/")
    print(f"Results : {RESULTS_DIR}/")

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    args = parser.parse_args()
    train(resume=args.resume)