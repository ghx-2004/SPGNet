import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_tensor


def normalize_prediction(pred):
    return (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)


def eval_mae(pred, gt, cuda=True):
    with torch.no_grad():
        if not torch.is_tensor(pred):
            pred = to_tensor(pred)
        if not torch.is_tensor(gt):
            gt = to_tensor(gt)
        if cuda:
            pred = pred.cuda()
            gt = gt.cuda()
        pred = normalize_prediction(pred)
        mae = torch.abs(pred.float() - gt.float()).mean()
        return mae.item()

def eval_Smeasure(pred, gt, cuda=True):
    alpha = 0.5
    with torch.no_grad():
        if not torch.is_tensor(pred):
            pred = to_tensor(pred)
        if not torch.is_tensor(gt):
            gt = to_tensor(gt)
        if cuda:
            pred = pred.cuda()
            gt = gt.cuda()
        pred = normalize_prediction(pred)
        gt = (gt >= 0.5).float()
        y = gt.mean()
        if y == 0:
            Q = 1.0 - pred.mean()
        elif y == 1:
            Q = pred.mean()
        else:
            Q = alpha * fun_S_object(pred, gt) + (1 - alpha) * fun_S_region(pred, gt)
        return Q.item() if isinstance(Q, torch.Tensor) else Q

def eval_Fmeasure(pred, gt, cuda=True):
    beta2 = 0.3
    with torch.no_grad():
        if not torch.is_tensor(pred):
            pred = to_tensor(pred)
        if not torch.is_tensor(gt):
            gt = to_tensor(gt)
        if cuda:
            pred = pred.cuda()
            gt = gt.cuda()
        pred = normalize_prediction(pred)
        prec, recall = fun_eval_pr(pred, gt, 255, cuda)
        
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall + 1e-8)
        f_score = f_score.nan_to_num(nan=0.0)
        return f_score.max().item()

def eval_Emeasure(pred, gt, cuda=True, eps=1e-8):
    if not torch.is_tensor(pred):
        pred = transforms.ToTensor()(pred)
    if not torch.is_tensor(gt):
        gt = transforms.ToTensor()(gt)
    if cuda:
        pred = pred.cuda()
        gt   = gt.cuda()
    pred = normalize_prediction(pred)
    dp = pred - pred.mean()
    dg = gt   - gt.mean()
    xi = 2 * (dp * dg) / (dp.pow(2) + dg.pow(2) + eps)
    enhanced = (xi + 1).pow(2) / 4.0
    E = enhanced.mean()
    return E.item()


def fun_eval_pr(y_pred, y, num, cuda=True):
    if cuda:
        prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
        thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    else:
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-8), tp / (y.sum() + 1e-8)
    return prec, recall


def fun_S_object(pred, gt):
    fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
    bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
    o_fg = fun_object(fg, gt)
    o_bg = fun_object(bg, 1 - gt)
    u = gt.mean()
    return u * o_fg + (1 - u) * o_bg

def fun_object(pred, gt):
    mask = gt > 0.5
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    temp = pred[mask]
    x = temp.mean()
    sigma_x = temp.std(unbiased=False)
    sigma_x = torch.nan_to_num(sigma_x, nan=0.0)
    return 2.0 * x / (x * x + 1.0 + sigma_x + 1e-8)

def fun_S_region(pred, gt):
    X, Y = fun_centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = fun_divideGT(gt, X, Y)
    p1, p2, p3, p4 = fun_dividePrediction(pred, X, Y)
    Q1 = fun_ssim(p1, gt1)
    Q2 = fun_ssim(p2, gt2)
    Q3 = fun_ssim(p3, gt3)
    Q4 = fun_ssim(p4, gt4)
    return w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4


def fun_centroid(gt):
    rows, cols = gt.size()[-2:]
    gt = gt.view(rows, cols)
    if gt.sum() == 0:
        X = torch.tensor(round(cols / 2)).long()
        Y = torch.tensor(round(rows / 2)).long()
    else:
        i = torch.arange(cols, device=gt.device).float()
        j = torch.arange(rows, device=gt.device).float()
        X = torch.round((gt.sum(dim=0) * i).sum() / gt.sum())
        Y = torch.round((gt.sum(dim=1) * j).sum() / gt.sum())
    return X.long(), Y.long()


def fun_divideGT(gt, X, Y):
    h, w = gt.size()[-2:]
    gt = gt.view(h, w)
    LT = gt[:Y, :X]
    RT = gt[:Y, X:]
    LB = gt[Y:, :X]
    RB = gt[Y:, X:]
    X = X.float()
    Y = Y.float()
    area = h * w
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4


def fun_dividePrediction(pred, X, Y):
    h, w = pred.size()[-2:]
    pred = pred.view(h, w)
    LT = pred[:Y, :X]
    RT = pred[:Y, X:]
    LB = pred[Y:, :X]
    RB = pred[Y:, X:]
    return LT, RT, LB, RB


def fun_ssim(pred, gt):
    h, w = pred.size()[-2:]
    N = h * w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x)**2).sum() / (N - 1 + 1e-8)
    sigma_y2 = ((gt - y)**2).sum() / (N - 1 + 1e-8)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-8)
    aplha = 4 * x * y * sigma_xy
    beta = (x*x + y*y) * (sigma_x2 + sigma_y2)
    if aplha != 0:
        Q = aplha / (beta + 1e-8)
    elif aplha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0.0
    return Q

class EvalThread:
    def __init__(self, loader, model=None, device=None, method="TMSOD", dataset="UVT2000", cuda=True, output_dir=None):
        self.loader = loader
        self.model = model
        self.device = device
        self.method = method
        self.dataset = dataset
        self.cuda = cuda
        self.logfile = os.path.join(output_dir, 'result.txt') if output_dir else None

    def run(self):
        mae = self.Eval_mae()
        sm = self.Eval_Smeasure()
        fm = self.Eval_fmeasure()
        return {'MAE': mae, 'S-measure': sm, 'F-measure': fm}

    def eval_with_model(self, image, thermal):
        self.model.eval()
        with torch.no_grad():
            image, thermal = image.to(self.device), thermal.to(self.device)
            pred = self.model(image, thermal)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            pred = F.interpolate(pred, size=image.shape[2:], mode='bilinear', align_corners=True)
            pred = pred.cpu().squeeze(0)  # (1, 1, H, W) → (H, W)
        return pred

    def Eval_mae(self):
        avg_mae, img_num = 0.0, 0
        with torch.no_grad():
            for image, gt, thermal in self.loader:
                pred = self.eval_with_model(image, thermal)
                mae = eval_mae(pred, gt, self.cuda)
                avg_mae += mae
                img_num += 1
        return avg_mae / img_num

    def Eval_Smeasure(self):
        avg_s, img_num = 0.0, 0
        with torch.no_grad():
            for image, gt, thermal in self.loader:
                pred = self.eval_with_model(image, thermal)
                sm = eval_Smeasure(pred, gt, self.cuda)
                avg_s += sm
                img_num += 1
        return avg_s / img_num

    def Eval_fmeasure(self):
        avg_f, img_num = 0.0, 0
        with torch.no_grad():
            for image, gt, thermal in self.loader:
                pred = self.eval_with_model(image, thermal)
                fm = eval_fmeasure(pred, gt, self.cuda)
                avg_f += fm
                img_num += 1
        return avg_f / img_num
