# your_dataset_module.py

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def apply_central_mask_with_block(pil_img, mask_size=64):
    """
    Takes a PIL image (assumed 128x128) and returns:
      - masked_img_pil: the image with a 64x64 center region replaced by black
      - block_img_pil: the cropped-out 64x64 center block
    """
    width, height = pil_img.size
    left = (width - mask_size) // 2
    top = (height - mask_size) // 2
    right = left + mask_size
    bottom = top + mask_size

    # Crop the center block
    block_img_pil = pil_img.crop((left, top, right, bottom))

    # Create a masked copy (fill center with black)
    masked_img_pil = pil_img.copy()
    black_square = Image.new("RGB", (mask_size, mask_size), (0, 0, 0))
    masked_img_pil.paste(black_square, (left, top, right, bottom))

    return masked_img_pil, block_img_pil

class InpaintingDataset(Dataset):
    """
    Loads images from 'root_dir', applies a central mask, and returns (masked_tensor, block_tensor).
    """
    def __init__(self, root_dir, transform=None, mask_size=64):
        self.root_dir = root_dir
        self.transform = transform
        self.mask_size = mask_size

        # Gather all image paths
        self.image_files = []
        for root, dirs, files in os.walk(self.root_dir):
            for fname in files:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, fname))

        if len(self.image_files) == 0:
            print(f"Warning: No images found in {root_dir}!")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        pil_img = Image.open(img_path).convert('RGB')

        # Apply central mask (returns two PIL images)
        masked_pil, block_pil = apply_central_mask_with_block(
            pil_img, mask_size=self.mask_size
        )

        # If transform is provided, apply it
        if self.transform:
            masked_tensor = self.transform(masked_pil)
            block_tensor  = self.transform(block_pil)
        else:
            # or do default transforms
            to_tensor = transforms.ToTensor()
            masked_tensor = to_tensor(masked_pil)
            block_tensor  = to_tensor(block_pil)

        return masked_tensor, block_tensor

