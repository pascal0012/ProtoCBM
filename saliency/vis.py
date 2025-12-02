import torch
import matplotlib.pyplot as plt

from saliency.utils import norm_tensor

def plot_cam(im: torch.Tensor, cam: torch.Tensor, title: str = "", figsize=(8,3)):
    """Helper function to plot the cam images.

    Args:
        im (torch.Tensor): input image
        cam (torch.Tensor): cam map
        title (str, optional): Title of the plot. Defaults to "".
        figsize (tuple, optional): Shape for the figure. Defaults to (8,3).
    """
    plt.figure(figsize=figsize)
    
    plt.subplot(1,3,1)
    plot_im = im.detach().cpu().squeeze().permute(1,2,0).numpy()
    # transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
    # unnormalize plot im 
    plot_im = norm_tensor(plot_im)
    plt.imshow(plot_im)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(cam)
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(plot_im)
    norm_grad = norm_tensor(cam)
    plt.imshow(norm_grad, alpha=norm_grad)
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
