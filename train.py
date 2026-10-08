import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.models as models
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import h5py
import numpy as np
import os, glob, random, math, io
import torch.nn.functional as F
from PIL import Image

# Resolution constant — change here to update everywhere
RES = 320

# ==============================================================================
# 1. AUGMENTATION SUITE — V29 PIPELINE
# ==============================================================================

def step_blue_floor(img_tensor):
    bias = torch.zeros_like(img_tensor)
    bias[0] = random.uniform(0.00, 0.02)
    bias[1] = random.uniform(0.01, 0.04)
    bias[2] = random.uniform(0.06, 0.10)
    return torch.max(img_tensor, bias)

def step_structural_debris(img_tensor):
    c, h, w = img_tensor.shape
    mask = torch.zeros((1, h, w), device=img_tensor.device)
    for _ in range(random.randint(1, 3)):
        cw, ch = random.randint(1, 3), random.randint(40, 120)
        if random.random() > 0.5: cw, ch = ch, cw
        cx = random.randint(0, max(0, w - cw))
        cy = random.randint(0, max(0, h - ch))
        beam = torch.rand(1, ch, cw, device=img_tensor.device) * 0.3 + 0.7
        mask[:, cy:cy+ch, cx:cx+cw] = torch.maximum(mask[:, cy:cy+ch, cx:cx+cw], beam)
    return torch.clamp(img_tensor + mask.repeat(3,1,1) * random.uniform(0.3, 0.8), 0.0, 1.0)

def step_edge_bright(img_tensor):
    k = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=torch.float32, device=img_tensor.device) / 16.0
    k = k.view(1,1,3,3).repeat(3,1,1,1)
    padded  = F.pad(img_tensor.unsqueeze(0), (1,1,1,1), mode='reflect')
    blurred = F.conv2d(padded, k, groups=3).squeeze(0)
    edges   = img_tensor - blurred
    bright_mask = (img_tensor.max(0, keepdim=True)[0] > 0.35).float()
    return torch.clamp(img_tensor + edges * bright_mask * random.uniform(0.4, 0.7), 0.0, 1.0)

def step_organic_noise(img_tensor):
    c, h, w = img_tensor.shape
    density = torch.rand((1, h//4, w//4), device=img_tensor.device)
    density = F.interpolate(density.unsqueeze(0), size=(h,w), mode='bilinear', align_corners=False).squeeze(0)
    sat_presence = img_tensor.max(0, keepdim=True)[0]
    void_mask    = (sat_presence < 0.2).float()
    salt_mask    = ((torch.rand((1,h,w), device=img_tensor.device) * density) > 0.98) & (void_mask > 0.5)
    n     = int(salt_mask.sum().item())
    noise = torch.zeros_like(img_tensor)
    if n > 0:
        cr    = torch.rand(n, device=img_tensor.device)
        r_val = torch.where(cr > 0.75, torch.rand(n, device=img_tensor.device) * 0.25, torch.zeros(n, device=img_tensor.device))
        noise[0, salt_mask[0]] = r_val
        noise[1, salt_mask[0]] = torch.rand(n, device=img_tensor.device) * 0.55
        noise[2, salt_mask[0]] = torch.rand(n, device=img_tensor.device) * 0.75
    return torch.clamp(img_tensor + noise, 0.0, 1.0)

def step_optical_smear(img_tensor):
    kx = random.choice([5, 7, 11])
    ky = random.choice([3, 5])
    if random.random() > 0.5: kx, ky = ky, kx
    glow = TF.gaussian_blur(img_tensor, [ky, kx], [ky/3.0, kx/3.0])
    return torch.clamp(img_tensor * 0.85 + glow * random.uniform(0.5, 0.9), 0.0, 1.0)

def step_vignette(img_tensor):
    c, h, w = img_tensor.shape
    y = torch.linspace(-1, 1, h, device=img_tensor.device).view(h, 1)
    x = torch.linspace(-1, 1, w, device=img_tensor.device).view(1, w)
    vig = torch.clamp(1.0 - torch.sqrt(x**2 + y**2) * random.uniform(0.10, 0.28), 0.55, 1.0)
    return torch.clamp(img_tensor * vig, 0.0, 1.0)

def step_chromatic_ab(img_tensor):
    shift = random.randint(1, 2)
    out   = img_tensor.clone()
    out[0] = torch.roll(img_tensor[0], -shift, dims=1) 
    out[2] = torch.roll(img_tensor[2],  shift, dims=1) 
    return out

def step_bg_bleed(img_tensor):
    c, h, w = img_tensor.shape
    y, x = torch.meshgrid(torch.arange(h, device=img_tensor.device), torch.arange(w, device=img_tensor.device), indexing='ij')
    angle = random.uniform(-45, 45)
    cx    = random.randint(w//4, 3*w//4)
    dist  = torch.abs((x.float() - cx) * math.cos(math.radians(angle)) - (y.float() - h//2) * math.sin(math.radians(angle)))
    streak    = torch.exp(-dist / 0.7) * random.uniform(0.05, 0.12)
    void_mask = (img_tensor.max(0)[0] < 0.12).float()
    out = img_tensor.clone()
    out[2] = torch.clamp(out[2] + streak * void_mask * 1.00, 0.0, 1.0)
    out[1] = torch.clamp(out[1] + streak * void_mask * 0.35, 0.0, 1.0)
    return out

def step_lens_flare(img_tensor):
    c, h, w = img_tensor.shape
    side = random.choice(['left', 'right', 'top'])
    if   side == 'left':  sx, sy = random.randint(0, w//6),      random.randint(0, h//3)
    elif side == 'right': sx, sy = random.randint(5*w//6, w-1),  random.randint(0, h//3)
    else:                 sx, sy = random.randint(w//4, 3*w//4), random.randint(0, h//8)
    y, x = torch.meshgrid(torch.arange(h, device=img_tensor.device), torch.arange(w, device=img_tensor.device), indexing='ij')
    flare = torch.zeros((1, h, w), device=img_tensor.device)
    for _ in range(random.randint(4, 12)):
        angle  = random.uniform(0, 2 * math.pi)
        length = random.uniform(40, 120)
        width  = random.uniform(0.4, 0.8)
        xr = (x.float()-sx)*math.cos(angle) - (y.float()-sy)*math.sin(angle)
        yr = (x.float()-sx)*math.sin(angle) + (y.float()-sy)*math.cos(angle)
        spear = torch.exp(-(torch.abs(xr)/length + torch.abs(yr)/width))
        flare[0] = torch.maximum(flare[0], spear * random.uniform(0.05, 0.30))
    flare_rgb = torch.cat([flare*0.25, flare*0.55, flare*1.00], dim=0)
    return torch.clamp(img_tensor + flare_rgb, 0.0, 1.0)

def step_secondary_light(img_tensor):
    c, h, w = img_tensor.shape
    cy = random.randint(h//4,  3*h//4)
    cx = random.randint(w//2,  w - 1)
    y, x = torch.meshgrid(torch.arange(h, device=img_tensor.device), torch.arange(w, device=img_tensor.device), indexing='ij')
    dist = torch.sqrt((x.float()-cx)**2 + (y.float()-cy)**2)
    glow = torch.exp(-dist / random.uniform(20, 40)) * random.uniform(0.08, 0.20)
    out  = img_tensor.clone()
    out[2] = torch.clamp(out[2] + glow, 0.0, 1.0)
    out[1] = torch.clamp(out[1] + glow * 0.45, 0.0, 1.0)
    return out

def step_jpeg(img_tensor):
    quality = random.randint(60, 78)
    np_img  = (img_tensor.permute(1,2,0).cpu().numpy() * 255).clip(0,255).astype(np.uint8)
    buf     = io.BytesIO()
    Image.fromarray(np_img).save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return torch.from_numpy(np.array(Image.open(buf)).astype(np.float32) / 255.0).permute(2,0,1).to(img_tensor.device).float()

# ==============================================================================
# 2. DATASET WITH V35 ROTATIONAL JITTER
# ==============================================================================

class SPADESDatasetV30Res320(Dataset):
    def __init__(self, h5_file, window_ms=400, is_train=True):
        self.h5_path  = h5_file
        self.window   = window_ms * 1000
        self.is_train = is_train
        with h5py.File(self.h5_path, 'r') as f:
            self.label_ts = f['labels/data']['timestamp'][:]
            self.tx = f['labels/data']['Tx'][:]
            self.ty = f['labels/data']['Ty'][:]
            self.tz = f['labels/data']['Tz'][:]
            self.qx = f['labels/data']['Qx'][:]
            self.qy = f['labels/data']['Qy'][:]
            self.qz = f['labels/data']['Qz'][:]
            self.qw = f['labels/data']['Qw'][:]
            self.length = len(self.label_ts)
            ps_samp = f['events/ps'][:500000]
            p_on    = np.mean(ps_samp)
            self.off_weight = p_on / (1.0 - p_on + 1e-6)
        self.f = None

    def __len__(self):
        return self.length

    def create_3c_tensor(self, xs, ys, ts, ps, target_ts, crop_box=None):
        img = np.zeros((3, RES, RES), dtype=np.float32)
        if len(xs) < 5:
            return torch.from_numpy(img).float(), 1.0
        weights = np.clip(1.0 - (target_ts - ts) / self.window, 0.2, 1.0)
        weights[ps == 0] *= self.off_weight
        if crop_box:
            cx, cy, sz = crop_box
            sz         = max(sz, 10.0)
            scale_hint = sz / 1280.0
            xi = np.clip((xs - (cx - sz/2)) * (RES/sz), 0, RES-1).astype(int)
            yi = np.clip((ys - (cy - sz/2)) * (RES/sz), 0, RES-1).astype(int)
        else:
            scale_hint = 1.0
            xi = np.clip(xs * ((RES-1)/1280), 0, RES-1).astype(int)
            yi = np.clip(ys * ((RES-1)/720), 0, RES-1).astype(int)
        t_step = (ts[-1] - ts[0]) / 3.0
        for i in range(3):
            m = (ts >= ts[0] + i*t_step) & (ts < ts[0] + (i+1)*t_step)
            if m.any():
                np.add.at(img[i], (yi[m], xi[m]), weights[m])
        tensor  = torch.from_numpy(img).unsqueeze(0)
        dilated = F.pad(F.max_pool2d(tensor, 2, 1, 0), (0,1,0,1))
        blur_k  = torch.ones((1,1,3,3)) / 9.0
        for ch in range(3):
            blurred = F.conv2d(dilated[:,ch:ch+1], blur_k, padding=1)
            dilated[:,ch:ch+1] = dilated[:,ch:ch+1] + 0.8*(dilated[:,ch:ch+1]-blurred)
        sharpened   = dilated.squeeze(0).numpy()
        log_img     = np.log1p(np.clip(sharpened, 0, None))
        sigmoid_img = 1.0 / (1.0 + np.exp(-10.0*(log_img/(np.max(log_img)+1e-9)-0.3)))
        final_img   = np.clip(sigmoid_img / (np.percentile(sigmoid_img, 99.5)+1e-8), 0, 1)
        return torch.from_numpy(final_img).float(), float(scale_hint)

    def apply_pipeline(self, img):
        if self.is_train and random.random() < 0.20:
            return step_jpeg(img)
        out = img
        out = step_blue_floor(out)
        out = step_edge_bright(out)
        out = step_organic_noise(out)
        out = step_vignette(out)
        out = step_chromatic_ab(out)
        if self.is_train:
            if random.random() < 0.20: out = step_structural_debris(out)
            if random.random() < 0.25: out = step_optical_smear(out)
            ev = random.random()
            if   ev < 0.15: out = step_lens_flare(out)
            elif ev < 0.35: out = step_secondary_light(out)
            elif ev < 0.50: out = step_bg_bleed(out)
            out = T.ColorJitter(brightness=0.15, contrast=0.15)(out)
        out = step_jpeg(out)
        return out

    def __getitem__(self, idx):
        if self.f is None:
            self.f = h5py.File(self.h5_path, 'r', swmr=True)

        target_ts = self.label_ts[idx]
        ev_ts     = self.f['events/ts']
        idx_e     = np.searchsorted(ev_ts, target_ts)
        idx_s     = np.searchsorted(ev_ts, target_ts - self.window)

        xs = self.f['events/xs'][idx_s:idx_e]
        ys = self.f['events/ys'][idx_s:idx_e]
        ts = ev_ts[idx_s:idx_e]
        ps = self.f['events/ps'][idx_s:idx_e]

        cx       = np.median(xs) if len(xs) > 0 else 640
        cy       = np.median(ys) if len(ys) > 0 else 360
        std_crop = np.clip(max(np.std(xs), np.std(ys))*12.0, 300, 1000) if len(xs) > 5 else 800

        if self.is_train and len(xs) > 0:
            cx += np.random.uniform(-std_crop*0.25, std_crop*0.25)
            cy += np.random.uniform(-std_crop*0.25, std_crop*0.25)
            if random.random() > 0.5:
                n_n = int(len(xs) * 0.40)
                xs  = np.concatenate([xs, np.random.uniform(0, 1280, n_n)])
                ys  = np.concatenate([ys, np.random.uniform(0, 720,  n_n)])
                ts  = np.concatenate([ts, np.random.uniform(ts.min(), ts.max(), n_n)])
                ps  = np.concatenate([ps, np.random.randint(0, 2, n_n)])

        img_f, _  = self.create_3c_tensor(xs, ys, ts, ps, target_ts)
        img_r, sc = self.create_3c_tensor(xs, ys, ts, ps, target_ts, crop_box=(cx, cy, std_crop))

        img_f = self.apply_pipeline(img_f)
        img_r = self.apply_pipeline(img_r)

        t_gt = torch.tensor([self.tx[idx], self.ty[idx], self.tz[idx]], dtype=torch.float32)
        q_gt = torch.tensor([self.qx[idx], self.qy[idx], self.qz[idx], self.qw[idx]], dtype=torch.float32)

        # ── V35 ROTATIONAL JITTER INTEGRATION ──
        if self.is_train and random.random() < 0.5:
            angle_deg = random.uniform(-45, 45)
            # Rotate images
            img_f = TF.rotate(img_f, angle_deg)
            img_r = TF.rotate(img_r, angle_deg)
            
            # Update Quaternion (Hamilton Product for Z-axis rotation)
            angle_rad = math.radians(angle_deg)
            qz = torch.tensor([0.0, 0.0, math.sin(angle_rad/2), math.cos(angle_rad/2)], dtype=torch.float32)
            
            x1, y1, z1, w1 = qz
            x2, y2, z2, w2 = q_gt
            q_gt = torch.tensor([
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
                w1*w2 - x1*x2 - y1*y2 - z1*z2
            ], dtype=torch.float32)

        return img_f, img_r, torch.tensor([sc]).float(), t_gt, q_gt

# ==============================================================================
# 3. V30 ARCHITECTURE
# ==============================================================================

class SparkV30Net(nn.Module):
    def __init__(self):
        super().__init__()
        effnet = models.efficientnet_v2_s(weights='DEFAULT')
        self.backbone_t = effnet.features
        self.t_head = nn.Sequential(
            nn.Linear(1280 + 1, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 3)
        )
        convnext = models.convnext_tiny(weights='DEFAULT')
        self.backbone_r = convnext.features
        self.r_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4)
        )

    def forward(self, x_f, x_r, scale):
        f_t = torch.flatten(nn.AdaptiveAvgPool2d(1)(self.backbone_t(x_f)), 1)
        t   = self.t_head(torch.cat([f_t, scale.view(-1,1).to(f_t.dtype)], dim=1))
        f_r   = torch.flatten(nn.AdaptiveAvgPool2d(1)(self.backbone_r(x_r)), 1)
        q_raw = self.r_head(f_r)
        q     = q_raw / (q_raw.norm(dim=1, keepdim=True) + 1e-8)
        return t, q

# ==============================================================================
# 4. LOSS & VALIDATION
# ==============================================================================

def spade_loss_v30(t_p, q_p, t_g, q_g):
    t_err = torch.mean(torch.norm(t_p - t_g, dim=1) / (torch.norm(t_g, dim=1) + 1e-8))
    dot   = torch.clamp(torch.abs(torch.sum(q_p * q_g, dim=1)), 0.0, 0.9999)
    r_err = torch.mean(2.0 * torch.acos(dot))
    return (1.0 * t_err) + (2.0 * r_err), t_err, r_err

def validate_model(model, loader, device):
    model.eval()
    val_t, val_r = [], []
    v_bias = torch.zeros((3, 320, 320), device=device)
    v_bias[1], v_bias[2] = 0.02, 0.08 
    
    with torch.no_grad():
        for img_f, img_r, sc, t_g, q_g in loader:
            img_f, img_r = img_f.to(device), img_r.to(device)
            sc, t_g, q_g = sc.to(device), t_g.to(device), q_g.to(device)
            img_f = torch.max(img_f, v_bias)
            img_r = torch.max(img_r, v_bias)
            
            with torch.amp.autocast('cuda'):
                t_p, q_p = model(img_f, img_r, sc)
                _, lt, lr = spade_loss_v30(t_p, q_p, t_g, q_g)
                
            val_t.append(lt.item())
            val_r.append(np.rad2deg(lr.item()))
    return np.mean(val_t), np.mean(val_r)

# ==============================================================================
# 5. TRAINING — BATCH 128 / 12 WORKERS
# ==============================================================================

if __name__ == "__main__":
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_h5  = sorted(glob.glob("data/RT*.h5"))
    random.seed(42)
    random.shuffle(all_h5)

    split      = int(len(all_h5) * 0.9)
    train_ds   = ConcatDataset([SPADESDatasetV30Res320(f, is_train=True)  for f in all_h5[:split]])
    val_ds     = ConcatDataset([SPADESDatasetV30Res320(f, is_train=False) for f in all_h5[split:]])
    
    # --- UPGRADED DATALOADER SETTINGS ---
    # Because 3-channel images use minimal RAM, we can push these to the limit.
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              num_workers=12, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    model     = SparkV30Net().to(device)
    # Scaled LR up to 3e-4 to account for the massive Batch 128
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler('cuda')

    ckpt_dir = "checkpoints_v30_res320"
    os.makedirs(ckpt_dir, exist_ok=True)
    existing    = glob.glob(f"{ckpt_dir}/spark_v30_res320_e*.pth")
    start_epoch = 1
    if existing:
        epochs      = [int(f.split('_e')[-1].split('.pth')[0]) for f in existing]
        latest      = max(epochs)
        latest_path = f"{ckpt_dir}/spark_v30_res320_e{latest}.pth"
        model.load_state_dict(torch.load(latest_path, map_location=device))
        start_epoch = latest + 1
        print(f"Resumed from {latest_path} -> starting epoch {start_epoch} at {RES}x{RES}")
    else:
        print(f"Training from scratch at {RES}x{RES} resolution.")

    for epoch in range(start_epoch, 15):
    # Emergency drop to 1e-5 starting at Epoch 7 to force final convergence
        if epoch < 7:
            current_lr = 5e-5
        else:
            current_lr = 1e-5
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr

        model.train()
        print(f"\n🚀 Epoch {epoch}/14 | LR: {current_lr:.0e} | Res: {RES}x{RES} | Batch: 128")

        for i, (img_f, img_r, sc, t_g, q_g) in enumerate(train_loader):
            img_f = img_f.to(device); img_r = img_r.to(device)
            sc    = sc.to(device);    t_g   = t_g.to(device);  q_g = q_g.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                t_p, q_p     = model(img_f, img_r, sc)
                loss, lt, lr = spade_loss_v30(t_p, q_p, t_g, q_g)

            if not torch.isnan(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                scaler.step(optimizer)
                scaler.update()

            if i % 100 == 0:
                print(f"  B{i:04d} | Loss:{loss:.4f}  T:{lt.item():.4f}  R:{np.rad2deg(lr.item()):.1f}°")

        vt, vr = validate_model(model, val_loader, device)
        print(f"  ── Val T:{vt:.4f}  R:{vr:.2f}° ──")
        torch.save(model.state_dict(), f"{ckpt_dir}/spark_v30_res320_e{epoch}.pth")