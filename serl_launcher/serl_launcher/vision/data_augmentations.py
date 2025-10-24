"""
Data augmentation functions for vision tasks in PyTorch.

Implements DrQ-style data augmentations and other image transforms.
"""

from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def random_crop(
    img: torch.Tensor,
    padding: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Random crop with edge padding.
    
    Args:
        img: Input image of shape [..., H, W, C]
        padding: Padding size
        generator: Random generator
        
    Returns:
        Randomly cropped image
    """
    *batch_dims, h, w, c = img.shape
    
    # Pad the image
    # PyTorch pad expects (left, right, top, bottom) for last 2 dims
    padded_img = F.pad(
        img.view(-1, h, w, c).permute(0, 3, 1, 2),  # (B, C, H, W)
        (padding, padding, padding, padding),
        mode='replicate'
    ).permute(0, 2, 3, 1)  # (B, H', W', C)
    
    # Random crop position
    if generator is None:
        generator = torch.Generator()
    
    crop_from_h = torch.randint(0, 2 * padding + 1, (1,), generator=generator).item()
    crop_from_w = torch.randint(0, 2 * padding + 1, (1,), generator=generator).item()
    
    # Crop
    cropped = padded_img[:, crop_from_h:crop_from_h+h, crop_from_w:crop_from_w+w, :]
    
    # Reshape back to original batch dims
    return cropped.view(*batch_dims, h, w, c)


def batched_random_crop(
    img: torch.Tensor,
    padding: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Apply random crop to a batch of images.
    
    Args:
        img: Batch of images of shape [B, H, W, C] or [B, T, H, W, C]
        padding: Padding size
        generator: Random generator
        
    Returns:
        Batch of randomly cropped images
    """
    if img.ndim == 4:
        # [B, H, W, C]
        return torch.stack([
            random_crop(img[i], padding, generator) 
            for i in range(img.shape[0])
        ])
    elif img.ndim == 5:
        # [B, T, H, W, C]
        result = []
        for i in range(img.shape[0]):
            batch_crops = torch.stack([
                random_crop(img[i, j], padding, generator)
                for j in range(img.shape[1])
            ])
            result.append(batch_crops)
        return torch.stack(result)
    else:
        return random_crop(img, padding, generator)


class GaussianBlur(nn.Module):
    """
    Gaussian blur augmentation.
    
    Args:
        kernel_size: Size of the Gaussian kernel
        sigma_min: Minimum sigma value
        sigma_max: Maximum sigma value
        apply_prob: Probability of applying the blur
    """
    
    def __init__(
        self,
        kernel_size: int = 5,
        sigma_min: float = 0.1,
        sigma_max: float = 2.0,
        apply_prob: float = 1.0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.apply_prob = apply_prob
    
    def forward(
        self,
        image: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Apply Gaussian blur.
        
        Args:
            image: Input image [..., H, W, C]
            generator: Random generator
            
        Returns:
            Blurred image
        """
        if generator is None:
            generator = torch.Generator()
        
        # Check if should apply
        if torch.rand(1, generator=generator).item() > self.apply_prob:
            return image
        
        # Sample sigma
        sigma = torch.rand(1, generator=generator).item()
        sigma = self.sigma_min + sigma * (self.sigma_max - self.sigma_min)
        
        # Convert to channels-first for blur
        original_shape = image.shape
        *batch_dims, h, w, c = image.shape
        image_flat = image.view(-1, h, w, c).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Apply Gaussian blur
        blurred = TF.gaussian_blur(image_flat, self.kernel_size, [sigma, sigma])
        
        # Convert back to channels-last
        blurred = blurred.permute(0, 2, 3, 1).view(*batch_dims, h, w, c)
        
        return blurred


def rgb_to_hsv(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert RGB to HSV.
    
    Args:
        image: RGB image [..., 3]
        
    Returns:
        Tuple of (H, S, V) tensors
    """
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    
    max_val = torch.maximum(torch.maximum(r, g), b)
    min_val = torch.minimum(torch.minimum(r, g), b)
    range_val = max_val - min_val
    
    # Value
    v = max_val
    
    # Saturation
    s = torch.where(max_val > 0, range_val / max_val, torch.zeros_like(max_val))
    
    # Hue
    norm = torch.where(range_val != 0, 1.0 / (6.0 * range_val), torch.ones_like(range_val) * 1e9)
    
    hr = norm * (g - b)
    hg = norm * (b - r) + 2.0 / 6.0
    hb = norm * (r - g) + 4.0 / 6.0
    
    h = torch.where(r == max_val, hr, torch.where(g == max_val, hg, hb))
    h = h * (range_val > 0).float()
    h = h + (h < 0).float()
    
    return h, s, v


def hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Convert HSV to RGB.
    
    Args:
        h: Hue [0, 1]
        s: Saturation [0, 1]
        v: Value [0, 1]
        
    Returns:
        RGB image [..., 3]
    """
    c = s * v
    m = v - c
    dh = (h % 1.0) * 6.0
    fmodu = dh % 2.0
    x = c * (1 - torch.abs(fmodu - 1))
    
    hcat = torch.floor(dh).long()
    
    rr = torch.where(
        (hcat == 0) | (hcat == 5), c,
        torch.where((hcat == 1) | (hcat == 4), x, torch.zeros_like(c))
    ) + m
    
    gg = torch.where(
        (hcat == 1) | (hcat == 2), c,
        torch.where((hcat == 0) | (hcat == 3), x, torch.zeros_like(c))
    ) + m
    
    bb = torch.where(
        (hcat == 3) | (hcat == 4), c,
        torch.where((hcat == 2) | (hcat == 5), x, torch.zeros_like(c))
    ) + m
    
    return torch.stack([rr, gg, bb], dim=-1)


class ColorJitter(nn.Module):
    """
    Color jittering augmentation.
    
    Args:
        brightness: Brightness jitter range
        contrast: Contrast jitter range
        saturation: Saturation jitter range
        hue: Hue jitter range
        to_grayscale_prob: Probability of converting to grayscale
        apply_prob: Probability of applying color jitter
    """
    
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        to_grayscale_prob: float = 0.1,
        apply_prob: float = 0.8,
    ):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.to_grayscale_prob = to_grayscale_prob
        self.apply_prob = apply_prob
    
    def forward(
        self,
        image: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Apply color jitter.
        
        Args:
            image: Input image [..., H, W, C]
            generator: Random generator
            
        Returns:
            Augmented image
        """
        if generator is None:
            generator = torch.Generator()
        
        # Check if should apply
        if torch.rand(1, generator=generator).item() > self.apply_prob:
            return image
        
        # Check grayscale
        if torch.rand(1, generator=generator).item() < self.to_grayscale_prob:
            # Convert to grayscale
            rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=image.device)
            grayscale = torch.sum(image * rgb_weights, dim=-1, keepdim=True)
            image = grayscale.repeat(*([1] * (image.ndim - 1)), 3)
            return torch.clamp(image, 0.0, 1.0)
        
        # Apply color transforms
        original_shape = image.shape
        *batch_dims, h, w, c = image.shape
        image_flat = image.view(-1, h, w, c)
        
        for img_idx in range(image_flat.shape[0]):
            img = image_flat[img_idx]
            
            # Random order of transforms
            order = torch.randperm(4, generator=generator)
            
            for idx in order:
                if idx == 0 and self.brightness > 0:
                    # Brightness
                    delta = torch.rand(1, generator=generator).item() * 2 * self.brightness - self.brightness
                    img = img + delta
                elif idx == 1 and self.contrast > 0:
                    # Contrast
                    factor = 1 + (torch.rand(1, generator=generator).item() * 2 - 1) * self.contrast
                    mean = img.mean(dim=(0, 1), keepdim=True)
                    img = factor * (img - mean) + mean
                elif idx == 2 and self.saturation > 0:
                    # Saturation
                    h, s, v = rgb_to_hsv(img)
                    factor = 1 + (torch.rand(1, generator=generator).item() * 2 - 1) * self.saturation
                    s = torch.clamp(s * factor, 0.0, 1.0)
                    img = hsv_to_rgb(h, s, v)
                elif idx == 3 and self.hue > 0:
                    # Hue
                    h, s, v = rgb_to_hsv(img)
                    delta = (torch.rand(1, generator=generator).item() * 2 - 1) * self.hue
                    h = (h + delta) % 1.0
                    img = hsv_to_rgb(h, s, v)
            
            image_flat[img_idx] = torch.clamp(img, 0.0, 1.0)
        
        return image_flat.view(*batch_dims, h, w, c)


class RandomFlip(nn.Module):
    """Random horizontal flip."""
    
    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob
    
    def forward(
        self,
        image: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Apply random horizontal flip.
        
        Args:
            image: Input image [..., H, W, C]
            generator: Random generator
            
        Returns:
            Flipped image
        """
        if generator is None:
            generator = torch.Generator()
        
        if torch.rand(1, generator=generator).item() < self.prob:
            # Flip along width dimension
            return torch.flip(image, dims=[-2])
        return image


class Solarize(nn.Module):
    """Solarization augmentation."""
    
    def __init__(self, threshold: float = 0.5, apply_prob: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.apply_prob = apply_prob
    
    def forward(
        self,
        image: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Apply solarization.
        
        Args:
            image: Input image [..., H, W, C]
            generator: Random generator
            
        Returns:
            Solarized image
        """
        if generator is None:
            generator = torch.Generator()
        
        if torch.rand(1, generator=generator).item() > self.apply_prob:
            return image
        
        return torch.where(image < self.threshold, image, 1.0 - image)


class DrQAugmentation(nn.Module):
    """
    DrQ-style data augmentation pipeline.
    
    Combines random crop and color jitter as used in DrQ algorithm.
    
    Args:
        pad: Padding size for random crop
        brightness: Brightness jitter
        contrast: Contrast jitter
        saturation: Saturation jitter
        hue: Hue jitter
    """
    
    def __init__(
        self,
        pad: int = 4,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
    ):
        super().__init__()
        self.pad = pad
        self.color_jitter = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
    
    def forward(
        self,
        image: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Apply DrQ augmentation.
        
        Args:
            image: Input image [B, H, W, C]
            generator: Random generator
            
        Returns:
            Augmented image
        """
        # Random crop
        image = batched_random_crop(image, self.pad, generator)
        
        # Color jitter
        image = self.color_jitter(image, generator)
        
        return image


# Convenience functions
def drq_image_augmentation(
    image: torch.Tensor,
    pad: int = 4,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Apply DrQ-style augmentation to image.
    
    Args:
        image: Input image [B, H, W, C]
        pad: Padding for random crop
        generator: Random generator
        
    Returns:
        Augmented image
    """
    aug = DrQAugmentation(pad=pad)
    return aug(image, generator)
