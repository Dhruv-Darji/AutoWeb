"""
Action Generation Module

SeeAct Action Generation Module: responsible for generating action plans based on input task data and history.

Input:
- input object with (screenshot, prompt)
- loaded model (Qwen-2.4-VL-2B-Instruct)
"""

import time
import re
from typing import Dict, Union

from AutoWeb.src.model_interface import VLModel
from AutoWeb.src.gpt_model import GPTVisionModel
from AutoWeb.src.logger import logger
from AutoWeb.src.config import get_seeact_image_detail


class SeeActActionGenerator:
    def __init__(self, model: Union[VLModel, GPTVisionModel]):
        self.model = model

    def _is_valid_dataset_action(self, text: str) -> bool:
        """Lightweight validator that checks whether `text` matches the
        Multimodal-Mind2Web / SeeAct one-line dataset-style formats.
        This does NOT rewrite/normalize the string — it only validates.
        """
        if not text or not isinstance(text, str):
            return False
        s = text.strip()
        # FINISH as exact standalone
        if re.fullmatch(r"FINISH", s, re.IGNORECASE):
            return True
        # TYPE: <value>
        if re.match(r"^(TYPE|type)\s*:\s*.+", s):
            return True
        # Bracketed form: [element] TEXT -> CLICK / -> TYPE: value / -> SELECT
        if re.search(r"\[.+\]\s+.+->\s*(CLICK|TYPE\s*:|SELECT)", s, re.IGNORECASE):
            return True
        return False

    def generate_plans(self, input_data: Dict) -> str:
        """
        Generate action plan based on the input data.
        
        Args:
            input_data: Dict containing prepared inputs for action generation, including:
            - "screenshot": website screenshot to be given in Action Generation
            - "prompt": the prompt text to be given in Action Generation (including instruction, history, etc.)
        
        Returns:
            Generated action plan as a dict containing `raw_text` and `output_text`.
            `output_text` is the model's own string (no automatic normalization). If
            the model output doesn't match the dataset-style format and we're
            using an API model, we attempt one focused retry asking the model to
            strictly follow the dataset format.
        """
        screenshot = input_data.get("screenshot", None)
        prompt = input_data.get("prompt", "")

        # Diagnostics: verify inputs before calling the model
        logger.info("  ⏳ Generating action plan using the model...")
        logger.debug(f"    - screenshot type: {type(screenshot)}, prompt length: {len(prompt) if prompt is not None else 0}")

        # Check that prompt contains an image placeholder the model recognizes (best-effort)
        try:
            preferred = getattr(self.model, "get_preferred_image_key", lambda _=None: "")
            pref_key = preferred(screenshot)
            if pref_key and pref_key not in prompt:
                logger.warning(f"    ⚠ prompt does not contain model-preferred image token '{pref_key}' (prompt may not be using correct placeholder)")
        except Exception:
            # non-fatal
            pass

        t0 = time.time()
        try:
            # For GPT API models, respect the SEEACT_IMAGE_DETAIL env var
            infer_kwargs = dict(
                image_or_tensor=screenshot,
                prompt_text=prompt,
                max_new_tokens=128,
                do_sample=False,
            )
            if isinstance(self.model, GPTVisionModel):
                infer_kwargs["image_detail"] = get_seeact_image_detail()
            generated_action_plan = self.model.infer(**infer_kwargs)
        except Exception as e:
            logger.exception(f"    ✗ Model inference failed: {e}")
            return {"error": str(e), "output_text": "", "raw_text": "", "latency": 0.0}

        # Extract model text (preserve as-is)
        if isinstance(generated_action_plan, dict):
            raw_text = generated_action_plan.get("raw_text") or generated_action_plan.get("output_text") or str(generated_action_plan)
        else:
            raw_text = str(generated_action_plan)

        output_text = (raw_text or "").strip()

        # If model output doesn't match required dataset format and we're
        # on an API model, give the model one explicit retry with a short
        # 'format-enforcement' instruction (do NOT rewrite the text here).
        if not self._is_valid_dataset_action(output_text) and isinstance(self.model, GPTVisionModel):
            logger.debug("    ⚠ model output did not match dataset format — retrying with stricter instruction")
            retry_prompt = (
                prompt
                + "\n\nIMPORTANT: Reply with EXACTLY one single-line action in the following dataset-style formats (and nothing else):\n"
                + "- [element_type] ELEMENT_TEXT -> CLICK\n- [element_type] ELEMENT_TEXT -> TYPE: <typed value>\n- [element_type] ELEMENT_TEXT -> SELECT\n- FINISH\n\nExamples:\n[heading]  CAR -> CLICK\n[combobox]  Enter pick up city, airport name, or airport code. -> TYPE: Brooklyn Central\n\nRespond ONLY with the single-line action (no quotes, code blocks or explanations)."
            )
            try:
                infer_kwargs["prompt_text"] = retry_prompt
                infer_kwargs["max_new_tokens"] = 64
                retry_out = self.model.infer(**infer_kwargs)
                if isinstance(retry_out, dict):
                    retry_text = retry_out.get("raw_text") or retry_out.get("output_text") or str(retry_out)
                else:
                    retry_text = str(retry_out)
                retry_text = (retry_text or "").strip()
                if self._is_valid_dataset_action(retry_text):
                    raw_text = retry_text
                    output_text = retry_text
                else:
                    logger.debug("    ⚠ retry also did not produce valid dataset-format output")
            except Exception as e:
                logger.exception(f"    ✗ Retry failed: {e}")

        t1 = time.time()
        latency = t1 - t0

        result = {"raw_text": raw_text, "output_text": output_text, "latency": latency}
        if isinstance(generated_action_plan, dict):
            merged = {**generated_action_plan, **result}
            # keep the model's own output_text if present, otherwise use raw_text
            merged["output_text"] = generated_action_plan.get("output_text") or output_text
            return merged
        return result