"""
SeeAct-Style Image Preprocessor

Minimal preprocessing as per SeeAct requirements:
- Resize image (keep aspect ratio)
- Normalize pixel values
- NO OCR
- NO DOM parsing (intentionally limited)

This keeps the baseline pure and defensible.
"""

from typing import Tuple, Optional, Dict
from PIL import Image
import numpy as np


class SeeActImagePreprocessor:
    """
    Minimal image preprocessing for SeeAct-style prediction.
    
    Key principles:
    1. Keep aspect ratio
    2. Minimal transformations
    3. No OCR or DOM (vision-only)
    """
    
    def __init__(self,
                 target_width: int = 1280,
                 target_height: int = 720,
                 keep_aspect_ratio: bool = True,
                 normalize: bool = False):
        """
        Args:
            target_width: Target width for resizing
            target_height: Target height for resizing
            keep_aspect_ratio: If True, maintain aspect ratio during resize
            normalize: If True, normalize pixel values to [0, 1]
        """
        self.target_width = target_width
        self.target_height = target_height
        self.keep_aspect_ratio = keep_aspect_ratio
        self.normalize = normalize
    
    def resize_with_aspect_ratio(self, 
                                 image: Image.Image,
                                 target_size: Tuple[int, int]) -> Tuple[Image.Image, Dict]:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: PIL Image
            target_size: (width, height)
        
        Returns:
            (resized_image, scale_info)
            
        scale_info contains:
            - original_width, original_height
            - new_width, new_height
            - scale_x, scale_y (for coordinate transformation)
            - padding_left, padding_top (if any)
        """
        orig_width, orig_height = image.size
        target_width, target_height = target_size
        
        if not self.keep_aspect_ratio:
            # Simple resize without maintaining aspect ratio
            resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            scale_info = {
                "original_width": orig_width,
                "original_height": orig_height,
                "new_width": target_width,
                "new_height": target_height,
                "scale_x": target_width / orig_width,
                "scale_y": target_height / orig_height,
                "padding_left": 0,
                "padding_top": 0
            }
            return resized, scale_info
        
        # Calculate scale to fit within target size
        scale = min(target_width / orig_width, target_height / orig_height)
        
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        # Resize
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # For SeeAct, we typically don't add padding - just return the resized image
        # If padding is needed, uncomment the code below:
        """
        # Create new image with padding if needed
        padded = Image.new('RGB', (target_width, target_height), (0, 0, 0))
        padding_left = (target_width - new_width) // 2
        padding_top = (target_height - new_height) // 2
        padded.paste(resized, (padding_left, padding_top))
        """
        
        scale_info = {
            "original_width": orig_width,
            "original_height": orig_height,
            "new_width": new_width,
            "new_height": new_height,
            "scale_x": scale,
            "scale_y": scale,
            "padding_left": 0,
            "padding_top": 0
        }
        
        return resized, scale_info
    
    def normalize_image(self, image: Image.Image) -> np.ndarray:
        """
        Convert PIL image to normalized numpy array.
        
        Args:
            image: PIL Image (RGB)
        
        Returns:
            numpy array with values in [0, 1] if normalize=True, else [0, 255]
        """
        arr = np.array(image)
        
        if self.normalize:
            arr = arr.astype(np.float32) / 255.0
        
        return arr
    
    def preprocess(self, image: Image.Image) -> Dict:
        """
        Preprocess image for SeeAct-style prediction.
        
        Args:
            image: PIL Image (RGB)
        
        Returns:
            {
                "image": PIL.Image (processed),
                "image_array": np.ndarray (optional, if normalize=True),
                "scale_info": dict (for coordinate transformation)
            }
        """
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Store original size
        original_size = image.size
        
        # Resize with aspect ratio
        resized_image, scale_info = self.resize_with_aspect_ratio(
            image,
            (self.target_width, self.target_height)
        )
        
        result = {
            "image": resized_image,
            "scale_info": scale_info,
            "original_size": original_size
        }
        
        # Optionally normalize
        if self.normalize:
            result["image_array"] = self.normalize_image(resized_image)
        
        return result
    
    def transform_coords_to_original(self,
                                    coords: Tuple[float, float],
                                    scale_info: Dict) -> Tuple[float, float]:
        """
        Transform coordinates from preprocessed image back to original image space.
        
        Args:
            coords: (x, y) in preprocessed image
            scale_info: Scale information from preprocess()
        
        Returns:
            (x, y) in original image coordinates
        """
        x, y = coords
        scale_x = scale_info["scale_x"]
        scale_y = scale_info["scale_y"]
        padding_left = scale_info.get("padding_left", 0)
        padding_top = scale_info.get("padding_top", 0)
        
        # Remove padding offset
        x = x - padding_left
        y = y - padding_top
        
        # Scale back to original
        orig_x = x / scale_x
        orig_y = y / scale_y
        
        return (orig_x, orig_y)
    
    def transform_bbox_to_original(self,
                                   bbox: Tuple[float, float, float, float],
                                   scale_info: Dict) -> Tuple[float, float, float, float]:
        """
        Transform bounding box from preprocessed image back to original image space.
        
        Args:
            bbox: (x, y, width, height) in preprocessed image
            scale_info: Scale information from preprocess()
        
        Returns:
            (x, y, width, height) in original image coordinates
        """
        x, y, w, h = bbox
        scale_x = scale_info["scale_x"]
        scale_y = scale_info["scale_y"]
        padding_left = scale_info.get("padding_left", 0)
        padding_top = scale_info.get("padding_top", 0)
        
        # Remove padding offset
        x = x - padding_left
        y = y - padding_top
        
        # Scale back to original
        orig_x = x / scale_x
        orig_y = y / scale_y
        orig_w = w / scale_x
        orig_h = h / scale_y
        
        return (orig_x, orig_y, orig_w, orig_h)


def preprocess_image_seeact(
    image: Image.Image,
    target_width: int = 1280,
    target_height: int = 720
) -> Dict:
    """
    Convenience function for SeeAct-style preprocessing.
    
    Args:
        image: PIL Image
        target_width: Target width
        target_height: Target height
    
    Returns:
        Preprocessed image dict with scale_info
    """
    preprocessor = SeeActImagePreprocessor(
        target_width=target_width,
        target_height=target_height,
        keep_aspect_ratio=True,
        normalize=False
    )
    return preprocessor.preprocess(image)
