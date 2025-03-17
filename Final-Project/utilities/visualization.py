# Author: Brett Kinsella and Rohan Gujral
# Date: 3/16/2025

from torchvision import transforms
import matplotlib.pyplot as plt

# Function to visualize center inpainting results
def visualize_center_inpainting(masked_imgs, generated_blocks, ground_truth, idx=0):

    masked_img = transforms.ToPILImage()(masked_imgs[idx].cpu())
    gen_img = transforms.ToPILImage()(generated_blocks[idx].cpu().detach())
    gt_img = transforms.ToPILImage()(ground_truth[idx].cpu())

    composite = masked_imgs[idx].clone().cpu()
    _, h, w = composite.shape
    block_size = 64
    
    start_h = (h - block_size) // 2
    start_w = (w - block_size) // 2
    composite[:, start_h:start_h+block_size, start_w:start_w+block_size] = generated_blocks[idx].cpu().detach()
    composite_img = transforms.ToPILImage()(composite)
    
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(masked_img)
    axs[0].set_title("Masked Input")
    axs[0].axis('off')

    axs[1].imshow(gt_img)
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')
    
    axs[2].imshow(gen_img)
    axs[2].set_title("Generated Block")
    axs[2].axis('off')

    axs[3].imshow(composite_img)
    axs[3].set_title("Composite Result")
    axs[3].axis('off')
    
    plt.tight_layout()
    plt.show()


# Function to visualize random block inpainting results
def visualize_random_block_inpainting(masked_imgs, original_imgs, generated_imgs, mask_tensors, idx=0):
    import matplotlib.pyplot as plt
    from torchvision import transforms
    
    masked_img = masked_imgs[idx].cpu()
    original_img = original_imgs[idx].cpu()
    generated_img = generated_imgs[idx].cpu().detach()
    mask = mask_tensors[idx].cpu()
    
    composite_img = original_img * (1 - mask) + generated_img * mask
    
    to_img = transforms.ToPILImage()
    masked_pil = to_img(masked_img)
    original_pil = to_img(original_img)
    generated_pil = to_img(generated_img)
    composite_pil = to_img(composite_img)
    
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(masked_pil)
    axs[0].set_title("Masked Input")
    axs[0].axis('off')
    
    axs[1].imshow(original_pil)
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')
    
    axs[2].imshow(generated_pil)
    axs[2].set_title("Generated Output")
    axs[2].axis('off')
    
    axs[3].imshow(composite_pil)
    axs[3].set_title("Composite Result")
    axs[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    