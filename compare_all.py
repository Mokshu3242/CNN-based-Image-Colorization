import os, cv2, csv, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import color
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import mean_squared_error as mse_fn
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE    = 96
L_NORM        = 50.0
AB_NORM       = 110.0
IMAGENET_MEAN = 0.449
IMAGENET_STD  = 0.226
RESULTS_DIR   = "results/results_comparison"
os.makedirs(RESULTS_DIR, exist_ok=True)

base_dir = Path("archive")
test_dir = base_dir / "test_images"

CHECKPOINTS = {
    "Simple CNN"   : "checkpoints/checkpoints_simplecnn/best.pth",
    "Simple U-Net" : "checkpoints/checkpoints_simpleunet/best.pth",
    "ResNet U-Net" : "checkpoints/checkpoints_resnet_unet/best.pth",
}

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 2, 1), nn.Tanh(),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU())
        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.output_layer = nn.Conv2d(64, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.tanh(self.output_layer(d1))

def dec_block(ic, oc):
    return nn.Sequential(
        nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(),
        nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU())

class ResNetUNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.enc1 = resnet.layer1; self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3; self.enc4 = resnet.layer4
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.dec4 = dec_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.dec3 = dec_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64,  2, stride=2); self.dec2 = dec_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64,  64,  2, stride=2); self.dec1 = dec_block(128, 64)
        self.up0 = nn.ConvTranspose2d(64,  32,  2, stride=2); self.dec0 = dec_block(32,  32)
        self.head = nn.Conv2d(32, 2, 1)
        self.tanh = nn.Tanh()
    def forward(self, x):
        x3 = (x.repeat(1, 3, 1, 1) + 1.0) / 2.0
        x3 = (x3 - IMAGENET_MEAN) / IMAGENET_STD
        e0 = self.enc0(x3); e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(e1);  e3 = self.enc3(e2); e4 = self.enc4(e3)
        d4 = self.dec4(torch.cat([self.up4(e4), e3], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e0], 1))
        return self.tanh(self.head(self.dec0(self.up0(d1))))

MODEL_CLASSES = {
    "Simple CNN"   : SimpleCNN,
    "Simple U-Net" : SimpleUNet,
    "ResNet U-Net" : ResNetUNet,
}

def load_all_models():
    loaded = {}
    for name, path in CHECKPOINTS.items():
        if not os.path.exists(path):
            print(f"SKIP {name} — checkpoint not found")
            continue
        model = MODEL_CLASSES[name]().to(DEVICE)
        ckpt  = torch.load(path, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        model.eval()
        print(f"Loaded {name} (epoch {ckpt.get('epoch','?')})")
        loaded[name] = model
    return loaded

def image_to_L_tensor(gray_96):
    rgb = np.stack([gray_96]*3, axis=2).astype(np.float32) / 255.0
    L   = color.rgb2lab(rgb).astype(np.float32)[:,:,0] / L_NORM - 1.0
    return torch.from_numpy(L).unsqueeze(0).unsqueeze(0).to(DEVICE)

def lab_to_rgb(L_np, AB_np):
    lab = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    lab[:,:,0]  = (L_np + 1.0) * L_NORM
    lab[:,:,1:] = AB_np.transpose(1, 2, 0) * AB_NORM
    return np.clip(color.lab2rgb(lab), 0, 1)

def smooth_upscale(AB_96, orig_h, orig_w):
    AB_hd = cv2.resize(AB_96, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    def smooth(ch):
        lo, hi = ch.min(), ch.max()
        if hi - lo < 1e-6: return ch
        n = ((ch - lo) / (hi - lo) * 255).astype(np.uint8)
        s = cv2.bilateralFilter(n, d=15, sigmaColor=20, sigmaSpace=20)
        return s.astype(np.float32) / 255.0 * (hi - lo) + lo
    return np.stack([smooth(AB_hd[:,:,0].astype(np.float32)),
                     smooth(AB_hd[:,:,1].astype(np.float32))], axis=2)

def colorize_single(image_path, loaded_models, saturation=1.4):
    img_bgr  = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Cannot read {image_path}"); return

    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    orig_h, orig_w = img_gray.shape

    gray_96  = cv2.resize(img_gray, (IMAGE_SIZE, IMAGE_SIZE))
    L_tensor = image_to_L_tensor(gray_96)
    L_hd     = color.rgb2lab(img_rgb.astype(np.float32)/255.0).astype(np.float32)[:,:,0]

    names  = list(loaded_models.keys())
    n_cols = 1 + len(names)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4))

    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title("Input (Grayscale)")
    axes[0].axis('off')

    for i, name in enumerate(names):
        with torch.no_grad():
            AB_pred = loaded_models[name](L_tensor)
        AB_96 = AB_pred[0].cpu().numpy().transpose(1, 2, 0) * AB_NORM
        AB_hd = smooth_upscale(AB_96, orig_h, orig_w) * saturation

        lab_out = np.zeros((orig_h, orig_w, 3), dtype=np.float32)
        lab_out[:,:,0]  = L_hd
        lab_out[:,:,1:] = AB_hd
        rgb_out = (np.clip(color.lab2rgb(lab_out), 0, 1) * 255).astype(np.uint8)

        axes[i+1].imshow(rgb_out)
        axes[i+1].set_title(name)
        axes[i+1].axis('off')

        stem = os.path.splitext(image_path)[0]
        out  = f"{stem}_{name.replace(' ','_').lower()}.png"
        cv2.imwrite(out, cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR))
        print(f"Saved : {out}")

    plt.suptitle("All Models — Side by Side", fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "single_image_comparison.png")
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Comparison saved : {out}")

def run_full_comparison(loaded_models):
    test_paths = list(test_dir.glob("*.*"))
    samples    = random.sample(test_paths, min(4, len(test_paths)))
    names      = list(loaded_models.keys())
    n_cols     = 2 + len(names)
    fig, axes  = plt.subplots(len(samples), n_cols, figsize=(n_cols*3, len(samples)*3))
    if len(samples) == 1: axes = axes[np.newaxis, :]

    for row, path in enumerate(samples):
        img_bgr = cv2.imread(str(path))
        img_rgb = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (IMAGE_SIZE, IMAGE_SIZE))
        img_f   = img_rgb.astype(np.float32) / 255.0
        lab     = color.rgb2lab(img_f).astype(np.float32)
        L_np    = lab[:,:,0] / L_NORM - 1.0
        AB_true = lab[:,:,1:] / AB_NORM
        L_t     = torch.from_numpy(L_np).unsqueeze(0).unsqueeze(0).to(DEVICE)

        axes[row,0].imshow((L_np+1.0)*L_NORM, cmap='gray', vmin=0, vmax=100)
        axes[row,0].set_title("Input" if row==0 else ""); axes[row,0].axis('off')
        axes[row,1].imshow(lab_to_rgb(L_np, AB_true.transpose(2,0,1)))
        axes[row,1].set_title("Ground Truth" if row==0 else ""); axes[row,1].axis('off')

        for col, name in enumerate(names):
            with torch.no_grad():
                AB_pred = loaded_models[name](L_t)
            axes[row, col+2].imshow(lab_to_rgb(L_np, AB_pred[0].cpu().numpy()))
            axes[row, col+2].set_title(name if row==0 else "")
            axes[row, col+2].axis('off')

    plt.suptitle("Model Comparison — Test Set", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "visual_comparison.png"), dpi=120, bbox_inches='tight')
    plt.close()
    print("Visual comparison saved")

    # metrics
    random.shuffle(test_paths)
    eval_paths = test_paths[:min(1600, len(test_paths))]
    results    = {n: {'mse':[], 'ssim':[]} for n in loaded_models}
    print(f"Evaluating on {len(eval_paths)} images...")

    for path in tqdm(eval_paths):
        img_bgr = cv2.imread(str(path))
        img_rgb = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (IMAGE_SIZE, IMAGE_SIZE))
        img_f   = img_rgb.astype(np.float32)/255.0
        lab     = color.rgb2lab(img_f).astype(np.float32)
        L_np    = lab[:,:,0]/L_NORM - 1.0
        AB_true = lab[:,:,1:]/AB_NORM
        L_t     = torch.from_numpy(L_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
        gt_rgb  = lab_to_rgb(L_np, AB_true.transpose(2,0,1))

        for name, model in loaded_models.items():
            with torch.no_grad():
                AB_pred = model(L_t)
            pred_rgb = lab_to_rgb(L_np, AB_pred[0].cpu().numpy())
            results[name]['mse'].append(mse_fn(gt_rgb, pred_rgb))
            results[name]['ssim'].append(ssim_fn(gt_rgb, pred_rgb, channel_axis=2, data_range=1.0))

    summary = {n: {'mse':  float(np.mean(results[n]['mse'])),
                   'ssim': float(np.mean(results[n]['ssim']))} for n in loaded_models}

    # bar charts
    names     = list(summary.keys())
    mse_vals  = [summary[n]['mse']  for n in names]
    ssim_vals = [summary[n]['ssim'] for n in names]
    x = np.arange(len(names))
    colors = ['#e74c3c','#3498db','#2ecc71']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    b1 = ax1.bar(x, mse_vals,  color=colors[:len(names)])
    b2 = ax2.bar(x, ssim_vals, color=colors[:len(names)])
    ax1.set_xticks(x); ax1.set_xticklabels(names)
    ax2.set_xticks(x); ax2.set_xticklabels(names)
    ax1.set_ylabel("MSE (lower is better)");  ax1.set_title("MSE Comparison")
    ax2.set_ylabel("SSIM (higher is better)"); ax2.set_title("SSIM Comparison")
    for bar, val in zip(b1, mse_vals):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0002,
                 f'{val:.5f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(b2, ssim_vals):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "metrics_comparison.png"), dpi=120, bbox_inches='tight')
    plt.close()
    print("Metrics chart saved")

    # loss overlay
    log_files = {
        "Simple CNN"   : "results/results_simplecnn/training_log.csv",
        "Simple U-Net" : "results/results_simpleunet/training_log.csv",
        "ResNet U-Net" : "results/results_resnet_unet/training_log.csv",
    }
    clr = {'Simple CNN':'#e74c3c', 'Simple U-Net':'#3498db', 'ResNet U-Net':'#2ecc71'}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, log_path in log_files.items():
        if not os.path.exists(log_path):
            print(f"No log for {name}, skipping."); continue
        epochs, train_l, val_l = [], [], []
        with open(log_path) as f:
            for row in csv.DictReader(f):
                epochs.append(int(row['epoch']))
                train_l.append(float(row['train_loss']))
                val_l.append(float(row['val_loss']))
        axes[0].plot(epochs, train_l, label=name, color=clr[name])
        axes[1].plot(epochs, val_l,   label=name, color=clr[name])
    for ax, title in zip(axes, ["Training Loss", "Validation Loss"]):
        ax.set_title(title); ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss"); ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle("Loss Curves — All Models", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_comparison.png"), dpi=120, bbox_inches='tight')
    plt.close()
    print("Loss overlay saved")

    # summary table
    print("\n" + "="*50)
    print(f"{'Model':<20} {'MSE':>10} {'SSIM':>10}")
    print("-"*50)
    for name, vals in summary.items():
        print(f"{name:<20} {vals['mse']:>10.5f} {vals['ssim']:>10.4f}")
    print("="*50)
    print(f"Best MSE  : {min(summary, key=lambda n: summary[n]['mse'])}")
    print(f"Best SSIM : {max(summary, key=lambda n: summary[n]['ssim'])}")
    print(f"\nAll outputs saved to '{RESULTS_DIR}/'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None,
                        help="Single image path — shows all 3 models side by side")
    parser.add_argument("--saturation", type=float, default=1.4)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    loaded = load_all_models()
    if not loaded:
        print("No checkpoints found."); exit(1)

    if args.image:
        if not os.path.exists(args.image):
            print(f"Image not found: {args.image}"); exit(1)
        colorize_single(args.image, loaded, args.saturation)
    else:
        run_full_comparison(loaded)