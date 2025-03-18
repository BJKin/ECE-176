# Author: Brett Kinsella and Rohan Gujral
# Date: 3/16/2025

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ssim, ms_ssim

# Class to evaluate the model's performance
class ModelEvaluator:
    def __init__(self, log_dir='runs'):
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
    def evaluate(self, generated, target, imageSize, epoch, global_step, prefix):
        mse_loss = F.mse_loss(generated, target)
        l1_loss = F.l1_loss(generated, target)
        
        psnr_val = 0
        for i in range(generated.size(0)):
            psnr_val += self.psnr(generated[i], target[i])
        psnr_val /= generated.size(0)
        
        if imageSize < 160:
            ssim_val = self.ssim(generated, target)
        else:
            ssim_val = self.ms_ssim(generated, target)
        
        metrics = {
            'L2 loss': mse_loss.item(),
            'L1 loss': l1_loss.item(),
            'Peak Signal-to-Noise Ratio': psnr_val.item(),
            'Structural Similarity Index': ssim_val.item()
        }
        
        for name, value in metrics.items():
             self.writer.add_scalar(f'{prefix}{name}', value, global_step)
        
        if epoch > self.last_image_epoch:
            self.writer.add_images(f'{prefix}Ground Truth', target[:5], epoch, dataformats='NCHW')
            self.writer.add_images(f'{prefix}Generated', generated[:5], epoch, dataformats='NCHW')
            self.writer.flush()
            self.last_image_epoch = epoch

        return metrics
    
    # Log training losses
    def log_losses(self, g_loss, d_loss, rec_loss, adv_loss, global_step, prefix):
        self.writer.add_scalar(f'{prefix}Generator', g_loss, global_step)
        if d_loss is not None:
            self.writer.add_scalar(f'{prefix}Discriminator', d_loss, global_step)
        self.writer.add_scalar(f'{prefix}Reconstruction', rec_loss, global_step)
        if adv_loss is not None:
            self.writer.add_scalar(f'{prefix}Adversarial', adv_loss, global_step)
        
    # Close the TensorBoard writer
    def close(self):
        self.writer.close()

    # Evaluate a model on validation or testing data
    def evaluate_data(self, generator, data_loader, device, criterion_rec, epoch=0, global_step=0, prefix="metrics/", discriminator=None, criterion_adv=None, lambda_rec=None, lambda_adv=None, is_random_block=False):
            generator.eval()
            if discriminator is not None:
                discriminator.eval()
                
            total_metrics = {
                'Reconstruction loss': 0.0,
                'Generator loss': 0.0,
                'Discriminator loss': 0.0,
                'Adversarial loss': 0.0,
                'Peak Signal-to-Noise Ratio': 0.0,
                'Structural Similarity Index': 0.0,
                'L1 loss': 0.0,
                'L2 loss': 0.0
            }
            
            last_samples = None
            batch_count = 0
            
            with torch.no_grad():
                for i, data in enumerate(data_loader):
                    if is_random_block:
                        masked_imgs, original_imgs, mask_tensors = data
                        masked_imgs = masked_imgs.to(device)
                        original_imgs = original_imgs.to(device)
                        mask_tensors = mask_tensors.to(device)
                        target = original_imgs
                        img_size = 227
                    else:
                        masked_imgs, blocks = data
                        masked_imgs = masked_imgs.to(device)
                        blocks = blocks.to(device)
                        target = blocks
                        img_size = 128
                    
                    # Generate output
                    output = generator(masked_imgs)
                    
                    # Calculate metrics
                    batch_metrics = self.evaluate(output, target, img_size, epoch, global_step + i, prefix)
                    
                    # Record all metrics
                    for key, value in batch_metrics.items():
                        if key in total_metrics:
                            total_metrics[key] += value
                    
                    # Calculate reconstruction loss
                    rec_loss = criterion_rec(output, target)
                    total_metrics['Reconstruction loss'] += rec_loss.item()
                    
                    # For GAN approach
                    if discriminator is not None and criterion_adv is not None:
                        # Real samples
                        real_outputs = discriminator(target)
                        real_labels = torch.ones_like(real_outputs, device=device)
                        d_loss_real = criterion_adv(real_outputs, real_labels)
                        
                        # Fake samples
                        fake_outputs = discriminator(output)
                        fake_labels = torch.zeros_like(fake_outputs, device=device)
                        d_loss_fake = criterion_adv(fake_outputs, fake_labels)
                        
                        # Combined losses
                        d_loss = d_loss_real + d_loss_fake
                        adv_loss = criterion_adv(fake_outputs, real_labels)
                        g_loss = lambda_rec * rec_loss + lambda_adv * adv_loss
                        
                        # Record GAN-specific metrics
                        total_metrics['Discriminator loss'] += d_loss.item()
                        total_metrics['Adversarial loss'] += adv_loss.item()
                        total_metrics['Generator loss'] += g_loss.item()
                        
                        # Log losses
                        self.log_losses(g_loss.item(), d_loss.item(), rec_loss.item(), adv_loss.item(), global_step + i, prefix)
                    else:
                        # For Reconstruction only approach
                        g_loss = rec_loss
                        total_metrics['Generator loss'] += g_loss.item()
                        self.log_losses(g_loss.item(), None, rec_loss.item(), None, global_step + i, prefix)
                    
                    # Save last batch for visualization
                    if i == len(data_loader) - 1:
                        if is_random_block:
                            last_samples = (masked_imgs, original_imgs, output, mask_tensors)
                        else:
                            last_samples = (masked_imgs, output, target)
                    
                    batch_count += 1
            
            # Calculate averages
            avg_metrics = {}
            for key in total_metrics:
                if batch_count > 0:
                    avg_metrics[key] = total_metrics[key] / batch_count
            
            return last_samples, avg_metrics


        