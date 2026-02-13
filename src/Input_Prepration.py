"""
SeeAct Input Preparation Module

Take values:
- single task action containing instruction, screenshot, cleaned HTML etc.
- history of previous actions for the task (if needed for input preparation)
And prepare the input for the Action Generation step. This include:
- website screenshot
- instruction (user given high level instruction)
- fixed prompt template (for now, can be more dynamic in the future)
- past history of actions

The output of this module will be the input for the Action Generation step, which will be a vision-language (Qwen-2.4-VL-2B-Instruct) model for SeeAct.
"""

from typing import Dict, List, Tuple, Optional

class SeeActInputPreparator:    
    def __init__(self):
        pass

    def prepare_input(self,
                      action: Dict,
                      history: List[Dict]) -> Dict:
        """
        Prepare input for a single action generation step.
        
        Args:
            action: Dict containing details of the current action (including instruction, website screenshot, cleaned HTML, etc.)
            history: List of previous Action Generation planes/steps in the task (if needed for input preparation)
        Returns:
            Dict containing prepared inputs for action generation, including:
            - "screenshot": website screenshot to be given in Action Generation
            - "prompt": the prompt text to be given in Action Generation (including instruction, history, etc.)
        """
        
        instruction = action.get("instruction", "")

        website_screenshot = action.get("screenshot", None)

        # If the screenshot is a PIL image and is larger than typical model input,
        # downscale it to a max size to avoid excessive patch/token counts.
        try:
            from PIL import Image
            if isinstance(website_screenshot, Image.Image):
                orig_size = website_screenshot.size
                max_w, max_h = (1280, 720)
                if orig_size[0] > max_w or orig_size[1] > max_h:
                    # compute high-quality resize that preserves more detail than aggressive thumbnailing
                    # increase cap to double quality (larger but still constrained)
                    # HIGH_QUALITY_MAX_W, HIGH_QUALITY_MAX_H = 4800, 2700  # doubled from 1600x900
                    HIGH_QUALITY_MAX_W, HIGH_QUALITY_MAX_H = 3200, 1800  # doubled from 1280x720
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
                    # Store image in action_id named temp file for further debugging in processed_image folder
                    import os
                    temp_dir = "processed_images"
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_path = os.path.join(temp_dir, f"{action.get('action_uid', 'unknown')}_input.jpg")
                    try:
                        website_screenshot.save(temp_path, quality=92, optimize=True)
                        print(f"    saved processed screenshot to {temp_path} (quality=92)")
                    except Exception:
                        website_screenshot.save(temp_path)
                        print(f"    saved processed screenshot to {temp_path}")
        except Exception:
            # non-fatal — continue with original image
            pass

        prompt_text = f"""
You are an expert web automation assistant. 
Your task is to generate textual action plans to accomplish the user's task based on the instruction and the website screenshot provided.

Instruction (What user want to achieve): {instruction},

Current website screenshot: <|vision_start|><|image_pad|><|vision_end|>

What previous actions planned and executed so far: {history}

Rules (IMPORTANT):
- Generate EXACTLY ONE next atomic UI action only.
- Do NOT produce multi-step trajectories or numbered lists.
- Do NOT repeat actions already present in "Previous Executed Actions".
- If no further action is required, output the single token: FINISH

Output format (one line only):
<ACTION_TYPE>: <ACTION_DETAIL>
Examples:
1. click: "Explore destinations for you"
2. scroll: "scroll upto "Customer Reviews" section"
3. input: "in search box, type 'wireless headphones'"
4. hover: "hover over the product image on the top right"
5. select: "in the date dropdown, select 'Next Week'"
FINISH

Based on the above instruction, website screenshot and history, please generate the NEXT SINGLE ACTION and nothing else.
"""
        return {
            "screenshot": website_screenshot,
            "prompt": prompt_text
        }