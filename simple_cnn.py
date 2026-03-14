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
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from skimage import color
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as calc_mse

IMAGE_SIZE = 96
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
L_NORM = 50.0   
AB_NORM = 110.0  

CHECKPOINT_DIR = "checkpoints/checkpoints_simplecnn"
RESULTS_DIR = "results/results_simplecnn"

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
        self.is_train    = is_train
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
            img     = np.array(img_pil)
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
    num_val    = max(1, int(0.1 * len(train_images)))
    val_imgs   = train_images[:num_val]
    train_imgs = train_images[num_val:] + unlabeled_imgs  
    random.shuffle(train_imgs)
    train_dataset = ColorizationDataset(train_imgs, is_train=True)
    val_dataset = ColorizationDataset(val_imgs,   is_train=False)
    test_dataset = ColorizationDataset(test_images, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches : {len(val_loader)}")
    print(f"Test batches : {len(test_loader)}")
    return train_loader, val_loader, test_loader

class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),   
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), 
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), 
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1), 
            nn.Tanh()                      
        )

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output

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
    avg_mse = float(np.mean(mse_list))
    avg_ssim = float(np.mean(ssim_list))

    return avg_mse, avg_ssim

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
    print(f"Saved sample grid: {save_path}")

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
    print("Simple CNN - Image Colorization")
    print(f"Device : {DEVICE}")
    print(f"Epochs : {NUM_EPOCHS}")
    print(f"Batch : {BATCH_SIZE}")
    print("=" * 50)
    train_loader, val_loader, _ = get_dataloaders()
    model = SimpleCNN().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}\n")
    start_epoch = 1
    best_val_loss = float('inf')
    if resume:
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler)
    log_path = os.path.join(RESULTS_DIR, "training_log.csv")
    log_file = open(log_path, 'a' if resume else 'w', newline='')
    log_csv = csv.writer(log_file)
    if not resume:
        log_csv.writerow(['epoch', 'train_loss', 'val_loss', 'val_mse', 'val_ssim', 'lr'])
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
            loss = criterion(AB_pred, AB_true)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        total_val_mse = 0.0
        total_val_ssim = 0.0
        saved_grid = False

        with torch.no_grad():
            for L, AB_true in tqdm(val_loader, desc=f"Epoch {epoch:03d} [Val]  ", leave=False):
                L = L.to(DEVICE)
                AB_true = AB_true.to(DEVICE)
                AB_pred = model(L)
                loss = criterion(AB_pred, AB_true)
                total_val_loss += loss.item()
                batch_mse, batch_ssim = compute_metrics(L, AB_pred, AB_true)
                total_val_mse  += batch_mse
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
            print(f"Learning rate reduced to {new_lr:.2e}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        save_checkpoint(epoch, model, optimizer, scheduler, val_loss, is_best)

        current_lr = optimizer.param_groups[0]['lr']
        log_csv.writerow([epoch, train_loss, val_loss, val_mse, val_ssim, current_lr])
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
              f"LR: {current_lr:.2e} | "
              f"{elapsed:.0f}s")

    log_file.close()

    epochs = list(range(1, len(all_train_losses) + 1))
    plt.figure(figsize=(9, 4))
    plt.plot(epochs, all_train_losses, label='Train Loss', color='steelblue')
    plt.plot(epochs, all_val_losses, label='Val Loss', color='tomato')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.title('Simple CNN - Training and Validation Loss')
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