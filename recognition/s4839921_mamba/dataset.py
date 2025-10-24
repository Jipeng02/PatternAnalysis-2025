from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T
import torch
class ColorizationDataset(Dataset):
    def __init__(self, img_dir, transform=None, max_samples=None):
        self.img_paths = [os.path.join(img_dir, f)
                          for f in os.listdir(img_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if max_samples is not None:
            self.img_paths = self.img_paths[:max_samples]
        self.transform = transform
        self.to_gray = T.Grayscale(num_output_channels=1)

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        color = self.transform(img) if self.transform else T.ToTensor()(img)
        gray  = self.to_gray(img)
        gray_tensor = self.transform(gray) if self.transform else T.ToTensor()(gray)
        gray_stacked = gray_tensor.repeat(3, 1, 1)  # Stack to make 3 channels
        return gray_stacked, color

class ColorJitter(object):
    """随机调整亮度、对比度、饱和度、色相"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, img):
        import random
        from PIL import ImageEnhance
        
        # Random brightness
        if self.brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor)
        
        # Random contrast
        if self.contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_factor)
        
        # Random saturation
        if self.saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(saturation_factor)
        
        # Random hue adjustment
        if self.hue > 0:
            import numpy as np
            from PIL import Image as PILImage
            import colorsys
            
            img_array = np.array(img).astype(np.float32) / 255.0
            h, s, v = [], [], []
            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    r, g, b = img_array[i, j]
                    h_val, s_val, v_val = colorsys.rgb_to_hsv(r, g, b)
                    h_val = (h_val + random.uniform(-self.hue, self.hue)) % 1.0
                    h.append(h_val)
                    s.append(s_val)
                    v.append(v_val)
            
            rgb_array = np.zeros_like(img_array)
            idx = 0
            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    r, g, b = colorsys.hsv_to_rgb(h[idx], s[idx], v[idx])
                    rgb_array[i, j] = [r, g, b]
                    idx += 1
            
            img = PILImage.fromarray((rgb_array * 255).astype(np.uint8))
        
        return img

# ==== Dataset with Augmentation ====
class AugColorizationDataset(Dataset):
    def __init__(self, img_dir, transform=None, max_samples=None, use_augmentation=True):
        self.img_paths = [os.path.join(img_dir, f)
                          for f in os.listdir(img_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if max_samples is not None:
            self.img_paths = self.img_paths[:max_samples]
        self.transform = transform
        self.to_gray = T.Grayscale(num_output_channels=1)
        
        self.use_augmentation = use_augmentation
        if use_augmentation:
            self.color_jitter = ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.4,
                hue=0.1
            )


    def __len__(self): 
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        
        # 教师扰动
        if self.use_augmentation and torch.rand(1).item() > 0.3:
            img = self.color_jitter(img)
        
        color = self.transform(img) if self.transform else T.ToTensor()(img)
        gray = self.to_gray(img)
        gray_tensor = self.transform(gray) if self.transform else T.ToTensor()(gray)
        gray_stacked = gray_tensor.repeat(3, 1, 1)
        return gray_stacked, color
