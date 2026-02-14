# If the screenshot is a PIL image and is larger than typical model input,
# downscale it to a max size to avoid excessive patch/token counts.

from PIL import Image

def downscale_image_if_needed(website_screenshot, max_w=1280, max_h=720):

    try:
        if isinstance(website_screenshot, Image.Image):
            orig_size = website_screenshot.size
            max_w, max_h = (1280, 720)
            if orig_size[0] > max_w or orig_size[1] > max_h:
                # compute high-quality resize that preserves more detail than aggressive thumbnailing
                # increase cap to double quality (larger but still constrained)                    
                HIGH_QUALITY_MAX_W, HIGH_QUALITY_MAX_H = 3200, 1800  # doubled from 3200x1800
                scale_w = HIGH_QUALITY_MAX_W / orig_size[0]
                scale_h = HIGH_QUALITY_MAX_H / orig_size[1]
                scale = min(scale_w, scale_h, 1.0)
                new_size = (max(1, int(orig_size[0] * scale)), max(1, int(orig_size[1] * scale)))

                # perform high-quality resize with LANCZOS (anti-aliased)
                website_screenshot = website_screenshot.resize(new_size, Image.Resampling.LANCZOS)
                print(f"[SeeActInputPreparator] resized screenshot {orig_size} -> {website_screenshot.size} (HQ x2)")
                # warn: larger images increase token/patch count and may slow inference or increase memory use
                if website_screenshot.size[0] * website_screenshot.size[1] > 1920 * 1080:
                    print("[SeeActInputPreparator] ⚠ using higher-quality image; this will increase tokenization size and GPU memory usage")
                
                return website_screenshot
                
    except Exception as e:
        print(f"[SeeActInputPreparator] error downscaling image: {e}")
        return website_screenshot