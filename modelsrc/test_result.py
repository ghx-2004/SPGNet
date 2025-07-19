#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
--------------------------------
1. Build a single-card model and load best_model.pth (automatically handle module.) "Prefix
2. Multiple cards are optional: load first and then package DataParallel
3. Force MSA4 window_size2=4 to ensure divisibility
4. Read all samples in numerical order, reason and save the binary graph
"""

import os
import re
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# --- parameter configuration --------------------------------------------------------------
RGB_ROOT      = '/home/gaohaixiao/Ourmodel/RGB_T/UVT2000/RGB'
THERMAL_ROOT  = '/home/gaohaixiao/Ourmodel/RGB_T/UVT2000/T'
WEIGHTS_PATH  = './train_best_model/best_model.pth'  # best_model.pth
SAVE_DIR      = ''
TEST_SIZE     = 384
THRESHOLD     = 0.5

from model import TMSOD

def sorted_numerical(file_list):
    def num_key(fname):
        num = re.findall(r'\d+', fname)
        return int(num[0]) if num else -1
    return sorted(file_list, key=num_key)

def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state[k[7:]] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=False)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TMSOD().to(device)

    load_checkpoint(model, WEIGHTS_PATH, device)

    # Multi-card support (optional)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    core = model.module if hasattr(model, 'module') else model
    core.MSA4_r.window_size2 = 4
    core.MSA4_t.window_size2 = 4

    model.eval()

    rgb_files = sorted_numerical([f for f in os.listdir(RGB_ROOT) if f.lower().endswith(('.jpg','.png'))])
    th_files  = sorted_numerical([f for f in os.listdir(THERMAL_ROOT) if f.lower().endswith(('.jpg','.png'))])
    assert len(rgb_files) == len(th_files), "RGB/T 文件数量不一致"

    img_tf = transforms.Compose([
        transforms.Resize((TEST_SIZE, TEST_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    th_tf = transforms.Compose([
        transforms.Resize((TEST_SIZE, TEST_SIZE)),
        transforms.ToTensor(),
    ])

    for idx, (rname, tname) in enumerate(tqdm(zip(rgb_files, th_files), total=len(rgb_files)), start=1):
        img = Image.open(os.path.join(RGB_ROOT, rname)).convert('RGB')
        th  = Image.open(os.path.join(THERMAL_ROOT, tname)).convert('L')

        x_img = img_tf(img).unsqueeze(0).to(device)
        x_th  = th_tf(th).unsqueeze(0).to(device)

        with torch.no_grad():
            out, _, _ = model(x_img, x_th)
            pred = torch.sigmoid(out)

        mask = (pred >= THRESHOLD).float().cpu().squeeze().numpy() * 255
        mask = mask.astype(np.uint8)

        save_path = os.path.join(SAVE_DIR, f"{idx}.png")
        Image.fromarray(mask).save(save_path)

    print(f"Success {len(rgb_files)} results were saved in {SAVE_DIR}")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
