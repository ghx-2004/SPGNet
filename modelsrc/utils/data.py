import math
import os
import random

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance


# ============================
# Data Augmentation
# ============================
def cv_random_flip(img, label, thermal):
    if random.randint(0, 1):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        thermal = thermal.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, thermal

def randomCrop(image, label, thermal):
    border = 30
    image_width, image_height = image.size
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    region = ((image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1,
              (image_width + crop_win_width) >> 1, (image_height + crop_win_height) >> 1)
    return image.crop(region), label.crop(region), thermal.crop(region)

def randomRotation(image, label, thermal):
    if random.random() > 0.8:
        angle = np.random.randint(-15, 15)
        image = image.rotate(angle, Image.BICUBIC)
        label = label.rotate(angle, Image.NEAREST)
        thermal = thermal.rotate(angle, Image.BICUBIC)
    return image, label, thermal

def colorEnhance(image):
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.5, 1.5))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.5, 1.5))
    image = ImageEnhance.Color(image).enhance(random.uniform(0.0, 2.0))
    image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.0, 3.0))
    return image

def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for _ in range(noiseNum):
        x = random.randint(0, img.shape[0] - 1)
        y = random.randint(0, img.shape[1] - 1)
        img[x, y] = 0 if random.randint(0, 1) == 0 else 255
    return Image.fromarray(img)
def randomScale(image, label, thermal, min_scale=0.5, max_scale=1.5):
    scale = random.uniform(min_scale, max_scale)
    new_size = (int(image.width * scale), int(image.height * scale))
    image = image.resize(new_size, Image.BILINEAR)
    label = label.resize(new_size, Image.NEAREST)
    thermal = thermal.resize(new_size, Image.BILINEAR)
    return image, label, thermal

def randomGaussian(image, mean=0.2, sigma=0.3):
    """
    Apply Gaussian noise to the image
    :param image:
    :param mean: mean value
    :param sigma: standard deviation
    :return:
    """
    gauss = np.random.normal(mean, sigma, image.size)
    gauss = gauss.reshape(image.shape).astype('uint8')
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255).astype('uint8')
    return Image.fromarray(noisy_image)

def randomErasing(image, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
    """
    random wipe
    :param image:
    :param sl: Minimum erasure area
    :param sh: Maximum erasure area
    :param r1: The aspect ratio of the erased area
    :param mean: The mean value of replacement
    :return:
    """
    img_h, img_w, img_c = image.shape
    for _ in range(100):
        area = img_h * img_w
        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)
        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))
        if img_h - h >= 0 and img_w - w >= 0:
            x1 = random.randint(0, img_h - h)
            y1 = random.randint(0, img_w - w)
            if img_c == 3:
                image[x1:x1 + h, y1:y1 + w, 0] = mean[0]
                image[x1:x1 + h, y1:y1 + w, 1] = mean[1]
                image[x1:x1 + h, y1:y1 + w, 2] = mean[2]
            else:
                image[x1:x1 + h, y1:y1 + w, 0] = mean[0]
            return Image.fromarray(image)
    return image

# ============================
# Dataset for Training
# ============================
class RGBTTMSODDataset(data.Dataset):
    def __init__(self, rgb_root, gt_root, thermal_root, trainsize, split='train'):
        """
        Args:
            rgb_root: RGB
            gt_root: GT
            thermal_root: Path of the heat map folder
            trainsize: target size
            split: Dataset partitioning, taking values of 'train', 'val' or 'test'
            They respectively represent the training set (70%), validation set (10%), and test set (20%)
        """
        self.trainsize = trainsize
        self.split = split

        self.images = sorted([os.path.join(rgb_root, f) for f in os.listdir(rgb_root) if f.endswith('.png') or f.endswith('.jpg')])
        self.gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')])
        self.thermals = sorted([os.path.join(thermal_root, f) for f in os.listdir(thermal_root)
                                if f.endswith('.jpg') or f.endswith('.png')])

        total_num = len(self.images)
        if self.split == 'train':
            indices = list(range(0, int(0.7 * total_num)))
        elif self.split == 'val':
            indices = list(range(int(0.7 * total_num), int(0.9 * total_num)))
        elif self.split == 'test':
            indices = list(range(int(0.9 * total_num), total_num))
        else:
            raise ValueError("Invalid split type. Choose from 'train', 'val', or 'test'.")

        self.images = [self.images[i] for i in indices]
        self.gts = [self.gts[i] for i in indices]
        self.thermals = [self.thermals[i] for i in indices]
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor()
        ])
        self.thermal_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        thermal = self.thermal_loader(self.thermals[index])

        image, gt, thermal = cv_random_flip(image, gt, thermal)
        image, gt, thermal = randomCrop(image, gt, thermal)
        image, gt, thermal = randomRotation(image, gt, thermal)
        image = colorEnhance(image)
        gt = randomPeper(gt)
        image, gt, thermal = randomScale(image, gt, thermal)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        thermal = self.thermal_transform(thermal)

        return image, gt, thermal

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def thermal_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def __len__(self):
        return self.size


# ============================
# Dataset for Testing
# ============================
class RGBTTMSODTestDataset:
    def __init__(self, rgb_root, gt_root, thermal_root, testsize):
        self.testsize = testsize
        self.images = sorted([os.path.join(rgb_root, f) for f in os.listdir(rgb_root)])
        self.gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root)])
        self.thermals = sorted([os.path.join(thermal_root, f) for f in os.listdir(thermal_root)])

        self.transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.thermal_transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor()])

        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        gt = self.binary_loader(self.gts[self.index])
        thermal = self.thermal_loader(self.thermals[self.index])

        name = os.path.basename(self.images[self.index]).replace('.jpg', '.png')
        image_post = image.resize(gt.size)

        image = self.transform(image).unsqueeze(0)
        thermal = self.thermal_transform(thermal).unsqueeze(0)

        self.index = (self.index + 1) % self.size
        return image, gt, thermal, name, np.array(image_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def thermal_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def __len__(self):
        return self.size

import torchvision.transforms as transforms
from torchvision.transforms import functional as F

class RGBTTMSODDataset_zzw(data.Dataset):
    def __init__(self, rgb_root, gt_root, thermal_root, trainsize, split='train'):
        self.trainsize = trainsize
        self.split = split

        self.images = sorted([os.path.join(rgb_root, f) for f in os.listdir(rgb_root) if f.endswith('.jpg')])
        self.gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')])
        self.thermals = sorted([os.path.join(thermal_root, f) for f in os.listdir(thermal_root)
                                if f.endswith('.jpg') or f.endswith('.png')])

        total_num = len(self.images)
        if self.split == 'train':
            indices = list(range(0, int(0.7 * total_num)))
        elif self.split == 'val':
            indices = list(range(int(0.7 * total_num), int(0.9 * total_num)))
        elif self.split == 'test':
            indices = list(range(int(0.9 * total_num), total_num))
        else:
            raise ValueError("Invalid split type. Choose from 'train', 'val', or 'test'.")

        self.images = [self.images[i] for i in indices]
        self.gts = [self.gts[i] for i in indices]
        self.thermals = [self.thermals[i] for i in indices]
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor()
        ])
        self.thermal_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        thermal = self.thermal_loader(self.thermals[index])

        if self.split == 'train':
            if random.random() > 0.5:
                image = F.hflip(image)
                gt = F.hflip(gt)
                thermal = F.hflip(thermal)

            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.trainsize, self.trainsize))
            image = F.crop(image, i, j, h, w)
            gt = F.crop(gt, i, j, h, w)
            thermal = F.crop(thermal, i, j, h, w)

            angle = random.randint(-15, 15)
            image = F.rotate(image, angle, interpolation=Image.BICUBIC)
            gt = F.rotate(gt, angle, interpolation=Image.NEAREST)
            thermal = F.rotate(thermal, angle, interpolation=Image.BICUBIC)

            scale = random.uniform(0.5, 1.5)
            new_size = (int(image.width * scale), int(image.height * scale))
            image = F.resize(image, new_size, interpolation=Image.BILINEAR)
            gt = F.resize(gt, new_size, interpolation=Image.NEAREST)
            thermal = F.resize(thermal, new_size, interpolation=Image.BILINEAR)

            image = F.adjust_brightness(image, random.uniform(0.5, 1.5))
            image = F.adjust_contrast(image, random.uniform(0.5, 1.5))
            image = F.adjust_saturation(image, random.uniform(0.0, 2.0))
            image = F.adjust_sharpness(image, random.uniform(0.0, 3.0))

            if random.random() > 0.5:
                image = np.array(image)
                noise = np.random.normal(0, 0.05, image.shape).astype(np.uint8)
                image = Image.fromarray(image + noise)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        thermal = self.thermal_transform(thermal)

        return image, gt, thermal

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def thermal_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def __len__(self):
        return self.size