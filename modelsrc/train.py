import csv
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from utils.GHX_eva_funcs import *

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多卡训练
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

from model import TMSOD
from utils.GHX_data import RGBTTMSODDataset
from utils.GHX_eva_funcs import EvalThread
from utils.GHX_smooth_loss import CombinedLoss, GHXSmoothnessLoss
from utils.scale_monitor import ScaleMonitor
from utils.utils import AverageMeter, adjust_lr, clip_gradient

dataset_name = 'VT5000'
train_root = f'/home/gaohaixiao/Ourmodel/RGB_T/{dataset_name}/RGB'
gt_root = f'/home/gaohaixiao/Ourmodel/RGB_T/{dataset_name}/GT'
thermal_root = f'/home/gaohaixiao/Ourmodel/RGB_T/{dataset_name}/T'
save_path = f'./train_result_{dataset_name}'
os.makedirs(save_path, exist_ok=True)
log_path = os.path.join(save_path, 'training_log_1.csv')

trainsize = 384
batchsize = 20
base_lr = 1e-5
weight_decay = 1e-4
grad_clip = 1.0
num_epochs = 100
decay_epoch = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# ========== data loading ==========
train_dataset = RGBTTMSODDataset(train_root, gt_root, thermal_root, trainsize, split='train')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

val_dataset = RGBTTMSODDataset(train_root, gt_root, thermal_root, trainsize, split='val')
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

# ========== Model & Optimization ==========
model = TMSOD().to(device)

# Load the pre-trained weights
swin_pretrained = "swin_base_patch4_window12_384_22k.pth"

if os.path.exists(swin_pretrained):
    print(f"Loading Swin pretrained weights from {swin_pretrained} ...")
    model.load_pre(swin_pretrained)
else:
    print("Warning: Swin pretrained weights not found!")

scale_monitor = ScaleMonitor(model, module_path='P_thermal', log_dir=save_path)

# ========== Modify the size of the dynamic window ==========
if hasattr(model, "module"):
    model.module.MSA4_r.window_size2 = 4
    model.module.MSA4_t.window_size2 = 4
else:
    model.MSA4_r.window_size2 = 4
    model.MSA4_t.window_size2 = 4

criterion = CombinedLoss(weight_dice=0.7, weight_bce=1.0).to(device)
optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=(base_lr*0.02))
scaler = GradScaler()

epoch_loss_meter = AverageMeter('TrainLoss')
train_consistency_meter = AverageMeter('TrainConsistency')

print("Start Training...")
best_mae = 1.0

# ========== Prepare for writing to the CSV log ==========
if not os.path.exists(log_path):
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'MAE', 'F-measure', 'S-measure', 'Consistency_Loss'])

for epoch in range(num_epochs):
    model.train()
    epoch_loss_meter.reset()
    train_consistency_meter.reset()
    optimizer.zero_grad()

    for i, (rgb, gt, thermal) in enumerate(train_loader):
        rgb, gt, thermal = rgb.to(device), gt.to(device), thermal.to(device)
        pred, p_saliency, P_thermal = model(rgb, thermal)
        loss = criterion(pred, gt)
        consistency_loss = model.consistency_loss()
        total_loss = loss + 0.1 * consistency_loss

        total_loss.backward()
        if (i + 1) % batchsize == 0:
            optimizer.step()
            optimizer.zero_grad()
        # scaler.scale(total_loss).backward()
        # scaler.unscale_(optimizer)
        # clip_gradient(optimizer, grad_clip)
        # scaler.step(optimizer)
        # scaler.update()

        epoch_loss_meter.update(loss.item(), rgb.size(0))
        train_consistency_meter.update(consistency_loss.item(), rgb.size(0))

    scheduler.step()

    # eval
    model.eval()
    val_loss_meter = AverageMeter('ValLoss')
    mae_meter = AverageMeter('MAE')
    f_measure_meter = AverageMeter('F-measure')
    s_measure_meter = AverageMeter('S-measure')
    e_measure_meter = AverageMeter('E-measure')

    with torch.no_grad():
        for i, (rgb, gt, thermal) in enumerate(val_loader):
            rgb, gt, thermal = rgb.to(device), gt.to(device), thermal.to(device)
            pred, _, _ = model(rgb, thermal)
            val_loss = criterion(pred, gt)
            val_loss_meter.update(val_loss.item(), rgb.size(0))

            pred = (pred > 0.5).float()
            mae = eval_mae(pred, gt, device)
            e_measure = eval_Emeasure(pred, gt, device)
            s_measure = eval_Smeasure(pred, gt, device)
            f_measure = eval_Fmeasure(pred, gt, device)

            mae_meter.update(mae, rgb.size(0))
            f_measure_meter.update(f_measure, rgb.size(0))
            s_measure_meter.update(s_measure, rgb.size(0))
            e_measure_meter.update(e_measure, rgb.size(0))

    # Save the best model
    torch.save(model.state_dict(), os.path.join(save_path, f"{epoch}_{mae_meter.avg:.4f}_model.pth"))
    if mae_meter.avg < best_mae:
        best_mae = mae_meter.avg
        torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))

    # Save the log
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, epoch_loss_meter.avg, val_loss_meter.avg, mae_meter.avg, f_measure_meter.avg, s_measure_meter.avg, train_consistency_meter.avg])

    print(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {epoch_loss_meter.avg:.4f}, Val Loss: {val_loss_meter.avg:.4f}, MAE: {mae_meter.avg:.4f}, F-measure: {f_measure_meter.avg:.4f}, S-measure: {s_measure_meter.avg:.4f}, E-measure: {e_measure_meter.avg:.4f} Consistency Loss: {train_consistency_meter.avg:.4f}")

