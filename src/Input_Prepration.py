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

        # # If the screenshot is a PIL image and is larger than typical model input,
        # # downscale it to a max size to avoid excessive patch/token counts.
        # try:
        #     from PIL import Image
        #     if isinstance(website_screenshot, Image.Image):
        #         orig_size = website_screenshot.size
        #         max_w, max_h = (1280, 720)
        #         if orig_size[0] > max_w or orig_size[1] > max_h:
        #             # compute high-quality resize that preserves more detail than aggressive thumbnailing
        #             # increase cap to double quality (larger but still constrained)                    
        #             HIGH_QUALITY_MAX_W, HIGH_QUALITY_MAX_H = 3200, 1800  # doubled from 3200x1800
        #             scale_w = HIGH_QUALITY_MAX_W / orig_size[0]
        #             scale_h = HIGH_QUALITY_MAX_H / orig_size[1]
        #             scale = min(scale_w, scale_h, 1.0)
        #             new_size = (max(1, int(orig_size[0] * scale)), max(1, int(orig_size[1] * scale)))

        #             # perform high-quality resize with LANCZOS (anti-aliased)
        #             website_screenshot = website_screenshot.resize(new_size, Image.Resampling.LANCZOS)
        #             print(f"[SeeActInputPreparator] resized screenshot {orig_size} -> {website_screenshot.size} (HQ x2)")
        #             # warn: larger images increase token/patch count and may slow inference or increase memory use
        #             if website_screenshot.size[0] * website_screenshot.size[1] > 1920 * 1080:
        #                 print("[SeeActInputPreparator] ⚠ using higher-quality image; this will increase tokenization size and GPU memory usage")
        #             # Store image in action_id named temp file for further debugging in processed_image folder
        #             import os
        #             temp_dir = "processed_images"
        #             os.makedirs(temp_dir, exist_ok=True)
        #             temp_path = os.path.join(temp_dir, f"{action.get('action_uid', 'unknown')}_input.jpg")
        #             try:
        #                 website_screenshot.save(temp_path, quality=92, optimize=True)
        #                 print(f"    saved processed screenshot to {temp_path} (quality=92)")
        #             except Exception:
        #                 website_screenshot.save(temp_path)
        #                 print(f"    saved processed screenshot to {temp_path}")
        # except Exception:
        #     # non-fatal — continue with original image
        #     pass

        # Format history into a numbered list for clearer context (handles list or string inputs)
        history_text = ""
        try:
            if isinstance(history, list):
                lines = []
                for i, h in enumerate(history):
                    # extract readable text from dict-like entries
                    if isinstance(h, dict):
                        text = h.get("action_plan") or h.get("output_text") or h.get("raw_text") or str(h)
                    else:
                        text = str(h)

                    text = "\n".join(text.strip().splitlines())
                    lines.append(f"{i}. {text}")
                history_text = "\n".join(lines) if lines else "(none)"
            elif isinstance(history, str):
                parts = [p.strip() for p in history.splitlines() if p.strip()]
                if parts:
                    history_text = "\n".join(f"{i}. {p}" for i, p in enumerate(parts))
                else:
                    history_text = history
            else:
                history_text = str(history)
        except Exception:
            history_text = str(history)

        prompt_text = f"""
You are an expert web automation assistant. Your job is to generate **exactly one atomic UI action** at a time that will help complete the user’s task on the current webpage. You should think like a human interacting with the page: observing, reasoning, and planning one small step at a time.

Do NOT produce multi-step plans, lists, code, or explanations — only one action.

────────────────────────────────────────────────────────────────────────────
INPUTS (do not repeat in output):

Instruction (goal): {instruction}
Current website screenshot:
<|vision_start|><|image_pad|><|vision_end|>

History of actions already taken:
{history_text}

────────────────────────────────────────────────────────────────────────────
THE ACTION SPACE (allowed actions):

1) click — Click on a specific UI element  
2) input — Type text into a field  
3) select — Choose an option from a dropdown  
4) finish — No more actions needed

All actions must follow this exact output format.

────────────────────────────────────────────────────────────────────────────
RESPONSE RULES (IMPORTANT):

• Generate **exactly ONE next action** that moves toward the goal.  
• Do NOT repeat an action already in history.  
• Do NOT make up UI element text — use what is visible.  
• Do NOT hallucinate or invent actions unrelated to the visible screenshot.  
• If the task is already complete or no further UI action is needed, output exactly `FINISH` (without quotes).  

────────────────────────────────────────────────────────────────────────────
OUTPUT FORMAT (one line only):
ACTION_TYPE: ACTION_DETAIL

• ACTION_TYPE must be exactly one of: click, input, select, FINISH  
• ACTION_DETAIL must clearly describe where and what to do on the UI

────────────────────────────────────────────────────────────────────────────
COMPLETE TASK EXAMPLES:

Task Goal: 
Given Textual plan for all steps (which is expected output of the Action Generation for single step at point):
1. CLICK : Click on heading "CAR" 
2. INPUT : Enter pick up city, airport name, or airport code and TYPE: "Brooklyn Central"
3. CLICK : Click on the suggestion "Brooklyn - Central (New York), US"
4. CLICK : Click on the "Pickup" textbox
5. CLICK : Click on the date "Sunday, April 9, 2023"
6. CLICK : Click on the date "Saturday, April 15, 2023"
7. CLICK : Click on the "Find cars button"

FINISH

INVALID OUTPUTS (don’t generate these):
• [] (empty value)
• click: #input-button
• input: search:nth-child(2)
• multi-step lists (e.g., “1. click…, 2. input…”)  
• explanations or thoughts in output  
• commands not executable as UI action  
• HTML, CSS selectors, code, or element ids

────────────────────────────────────────────────────────────────────────────
Now based on the instruction, the screenshot, and history, generate the **next single action** and nothing else.

"""
        return {
            "screenshot": website_screenshot,
            "prompt": prompt_text
        }