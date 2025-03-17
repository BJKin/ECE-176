# Author: Brett Kinsella and Rohan Gujral
# Date: 3/16/2025

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import torch
import numpy as np


# Function to resize images in a directory to specified sizes
def resize_images(source_dir, output_dir, target_sizes={'128x128': (128, 128), '227x227': (227, 227)}):
    os.makedirs(output_dir, exist_ok=True)
    
    for size_label, dims in target_sizes.items():
        size_dir = os.path.join(output_dir, size_label)
        os.makedirs(size_dir, exist_ok=True)
        print(f"Created directory for {size_label} images: {size_dir}")
    
    image_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images to resize")
    
    for i, file_path in enumerate(image_files):
        try:
            rel_path = os.path.relpath(file_path, source_dir)

            with Image.open(file_path) as img:
                img = img.convert('RGB')

                for size_label, dims in target_sizes.items():
                    target_dir = os.path.join(output_dir, size_label, os.path.dirname(rel_path))
                    os.makedirs(target_dir, exist_ok=True)
                    resized_img = img.resize(dims, Image.LANCZOS)
                    output_path = os.path.join(output_dir, size_label, rel_path)
                    resized_img.save(output_path)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("Resizing complete!")

# Function to apply a central mask and extract the block from the image
def apply_central_mask_with_block(pil_img, mask_size=64):
    width, height = pil_img.size
    left = (width - mask_size) // 2
    top = (height - mask_size) // 2
    right = left + mask_size
    bottom = top + mask_size

    block_img_pil = pil_img.crop((left, top, right, bottom))
    masked_img_pil = pil_img.copy()
    
    mean_color = (117, 104, 123)
    mean_square = Image.new("RGB", (mask_size, mask_size), mean_color)
    
    masked_img_pil.paste(mean_square, (left, top, right, bottom))

    return masked_img_pil, block_img_pil

# Function to apply random block masking 
def apply_random_block_mask(pil_img, block_size=32, coverage=0.25):
    width, height = pil_img.size
    masked_img_pil = pil_img.copy()
    
    mask_array = np.zeros((height, width), dtype=np.float32)
    img_area = width * height
    block_area = block_size * block_size
    num_blocks = int((coverage * img_area) / block_area)
    mean_color = (117, 104, 123)

    for _ in range(num_blocks):
        max_x = width - block_size
        max_y = height - block_size
            
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        mask = Image.new("RGB", (block_size, block_size), mean_color)
        masked_img_pil.paste(mask, (x, y))
        mask_array[y:y+block_size, x:x+block_size] = 1.0

    return masked_img_pil, pil_img, mask_array

# Dataset class for inpainting task
class CenterInpainting(Dataset):
    def __init__(self, root_dir, mask_size=64):
        self.root_dir = root_dir
        self.mask_size = mask_size

        self.image_files = []
        for root, dirs, files in os.walk(self.root_dir):
            for fname in files:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, fname))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        pil_img = Image.open(img_path).convert('RGB')

        masked_pil, block_pil = apply_central_mask_with_block(
            pil_img, mask_size=self.mask_size
        )

        full_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        block_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        masked_tensor = full_transform(masked_pil)
        block_tensor = block_transform(block_pil)

        return masked_tensor, block_tensor


# Class for random block inpainting
class RandomBlockInpainting(Dataset):
    def __init__(self, root_dir, block_size=32, coverage=0.25):
        self.root_dir = root_dir
        self.block_size = block_size
        self.coverage = coverage

        self.image_files = []
        for root, dirs, files in os.walk(self.root_dir):
            for fname in files:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, fname))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        pil_img = Image.open(img_path).convert('RGB')

        masked_pil, original_pil, mask_array = apply_random_block_mask(
            pil_img, block_size=self.block_size, coverage=self.coverage
        )

        transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor()
        ])

        masked_tensor = transform(masked_pil)
        original_tensor = transform(original_pil)
        
        mask_tensor = torch.from_numpy(mask_array)
        mask_tensor = transforms.Resize((227, 227))(mask_tensor.unsqueeze(0)).squeeze(0)
        
        mask_tensor = mask_tensor.unsqueeze(0).repeat(3, 1, 1)
        
        return masked_tensor, original_tensor, mask_tensor