"""
Action Generation Module

SeeAct Action Generation Module: responsible for generating action plans based on input task data and history.

Input:
- input object with (screenshot, prompt)
- loaded model (Qwen-2.4-VL-2B-Instruct)
"""

import time
from typing import Dict

from AutoWeb.src.model_interface import VLModel


class SeeActActionGenerator:
    def __init__(self, model:VLModel):
        self.model = model

    def generate_plans(self, input_data: Dict) -> str:
        """
        Generate action plan based on the input data.
        
        Args:
            input_data: Dict containing prepared inputs for action generation, including:
            - "screenshot": website screenshot to be given in Action Generation
            - "prompt": the prompt text to be given in Action Generation (including instruction, history, etc.)
        
        Returns:
            Generated action plan as a string.
        """
        screenshot = input_data.get("screenshot", None)
        prompt = input_data.get("prompt", "")

        # Here we would have the actual code to feed the prompt and screenshot into the model and get the generated action plan.
        # For now, we will just return a placeholder string.
        # Diagnostics: verify inputs before calling the model
        print("  ⏳ Generating action plan using the model...")
        print(f"    - screenshot type: {type(screenshot)}, prompt length: {len(prompt) if prompt is not None else 0}")

        # Short prompt preview + stable hash so we can compare notebook vs pipeline
        try:
            preview = (prompt or "")[:400].replace("\n", " ")
            import hashlib
            prompt_hash = hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()[:10]
            print(f"    - prompt preview: {preview}")
            print(f"    - prompt hash: {prompt_hash}")
        except Exception:
            pass

        # Check that prompt contains an image placeholder the model recognizes (best-effort)
        try:
            preferred = getattr(self.model, "get_preferred_image_key", lambda _=None: "")
            pref_key = preferred(screenshot)
            if pref_key and pref_key not in prompt:
                print(f"    ⚠ prompt does not contain model-preferred image token '{pref_key}' (prompt may not be using correct placeholder)")
        except Exception:
            # non-fatal
            pass

        try:
            t0 = time.time()
            generated_action_plan = self.model.infer(
                image_or_tensor=screenshot,
                prompt_text=prompt,
                max_new_tokens= 256,
                do_sample=False
            )
            t1 = time.time()
            print(f"    ✓ model.infer() returned in {t1 - t0:.2f}s (used max_new_tokens={256})")
        except Exception as e:
            print(f"    ✗ Model inference failed: {e}")
            return {"error": str(e), "raw_text": "", "latency": 0.0}

        # Normalize return shape: if model.infer returns dict, keep as-is; if it returned string, wrap it
        if isinstance(generated_action_plan, dict):
            return {**generated_action_plan, "latency": t1 - t0}
        else:
            return {"raw_text": str(generated_action_plan), "latency": t1 - t0}