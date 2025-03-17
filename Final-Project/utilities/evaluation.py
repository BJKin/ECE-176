# Author: Brett Kinsella and Rohan Gujral
# Date: 3/16/2025

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ssim, ms_ssim

# Class to evaluate the model's performance
class ModelEvaluator:
    def __init__(self, log_dir='runs/metrics'):
        self.writer = SummaryWriter(log_dir)
        self.last_image_epoch = -1

    # Calculate peak signal-to-noise ratio
    def psnr(self, generated, target):
        mse = F.mse_loss(generated, target)
        psnr = 10 * torch.log10(1.0 / mse)
        return psnr
    
    # Calculate structural similarity index (SSIM)
    def ssim(self, generated, target):
        if generated.min() < 0 or target.min() < 0:
            generated_norm = (generated + 1) / 2
            target_norm = (target + 1) / 2
        else:
            generated_norm = generated / generated.max()
            target_norm = target / target.max()
        
        ssim_val = ssim(
            generated_norm, 
            target_norm, 
            data_range=1.0, 
            size_average=True
        )
        
        return ssim_val
    
    # Calculate multi-scale SSIM
    def ms_ssim(self, generated, target):
        if generated.min() < 0 or target.min() < 0:
            generated_norm = (generated + 1) / 2
            target_norm = (target + 1) / 2
        else:
            generated_norm = generated / generated.max()
            target_norm = target / target.max()
        
        ms_ssim_val = ms_ssim(
            generated_norm, 
            target_norm, 
            data_range=1.0, 
            size_average=True 
        )
        return ms_ssim_val

    # Evaluate a batch and log metrics to TensorBoard
    def evaluate(self, generated, target, imageSize, epoch, global_step):
        mse_loss = F.mse_loss(generated, target)
        l1_loss = F.l1_loss(generated, target)
        
        psnr_val = 0
        for i in range(generated.size(0)):
            psnr_val += self.psnr(generated[i], target[i])
        psnr_val /= generated.size(0)
        
        if imageSize < 160:
            ssim_val = self.ssim(generated, target)
        else:
            ssim_val=self.ms_ssim(generated, target)
        
        metrics = {
            'mse_loss': mse_loss.item(),
            'l1_loss': l1_loss.item(),
            'psnr': psnr_val.item(),
            'ssim': ssim_val.item()
        }
        
        for name, value in metrics.items():
            self.writer.add_scalar(f'metrics/{name}', value, global_step)
        
        if epoch > self.last_image_epoch:
            self.writer.add_images('target', target[:10], epoch, dataformats='NCHW')
            self.writer.add_images('generated', generated[:10], epoch, dataformats='NCHW')
            self.writer.flush()
            self.last_image_epoch = epoch
    
    # Log training losses
    def log_losses(self, g_loss, d_loss, rec_loss, adv_loss, epoch, global_step):
        self.writer.add_scalar('loss/generator', g_loss, global_step)
        self.writer.add_scalar('loss/discriminator', d_loss, global_step)
        self.writer.add_scalar('loss/reconstruction', rec_loss, global_step)
        self.writer.add_scalar('loss/adversarial', adv_loss, global_step)
        
    # Close the TensorBoard writer
    def close(self):
        self.writer.close()