"""
Action Generation Module

SeeAct Action Generation Module: responsible for generating action plans based on input task data and history.

Input:
- input object with (screenshot, prompt)
- loaded model (Qwen-2.4-VL-2B-Instruct)
"""

import time
import re
import json
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

    def _parse_structured_output(self, raw_text: str) -> Dict:
        """Parse structured JSON output from model (PolicyHub mode).

        Fallback chain:
        1. ``json.loads()`` on full text
        2. Regex-extract a JSON block from surrounding prose
        3. Regex-extract individual ``action``, ``confidence``, ``hitl_reason`` fields
        4. Treat entire output as plain action text with ``confidence=50``

        Returns:
            ``{"action": str, "confidence": int, "hitl_reason": str}``
        """
        if not raw_text or not isinstance(raw_text, str):
            return {"action": "", "confidence": 50, "hitl_reason": "empty model output"}

        text = raw_text.strip()

        # --- 1. Direct JSON parse ---
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "action" in obj:
                return {
                    "action": str(obj.get("action", "")).strip(),
                    "confidence": int(obj.get("confidence", 50)),
                    "hitl_reason": str(obj.get("hitl_reason", "")).strip(),
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # --- 2. Extract JSON block from surrounding text ---
        json_match = re.search(r'(\{[^{}]*"action"[^{}]*\})', text, re.DOTALL)
        if json_match:
            try:
                obj = json.loads(json_match.group(1))
                if isinstance(obj, dict) and "action" in obj:
                    return {
                        "action": str(obj.get("action", "")).strip(),
                        "confidence": int(obj.get("confidence", 50)),
                        "hitl_reason": str(obj.get("hitl_reason", "")).strip(),
                    }
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # --- 3. Regex field extraction ---
        action_m = re.search(r'"action"\s*:\s*"([^"]+)"', text)
        conf_m = re.search(r'"confidence"\s*:\s*(\d+)', text)
        reason_m = re.search(r'"hitl_reason"\s*:\s*"([^"]*)"', text)
        if action_m:
            return {
                "action": action_m.group(1).strip(),
                "confidence": int(conf_m.group(1)) if conf_m else 50,
                "hitl_reason": reason_m.group(1).strip() if reason_m else "",
            }

        # --- 4. Fallback: treat entire text as action ---
        logger.debug("    [_parse_structured_output] Could not parse JSON — falling back to plain text")
        return {
            "action": text,
            "confidence": 50,
            "hitl_reason": "model did not produce structured JSON output",
        }

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

        # --- Parse structured JSON output (PolicyHub mode) ---
        # Try to extract {action, confidence, hitl_reason} from model output.
        # If parsing succeeds, use the extracted action as output_text and
        # carry confidence/hitl_reason forward.  If it fails (e.g. legacy
        # prompt without PolicyHub), confidence defaults to 50.
        parsed = self._parse_structured_output(output_text)
        parsed_action = parsed.get("action", "")
        confidence = parsed.get("confidence", 50)
        policy_risk = parsed.get("policy_risk", False)
        hitl_reason = parsed.get("hitl_reason", "")

        # If structured parse yielded a valid action, prefer it as output_text
        if parsed_action and self._is_valid_dataset_action(parsed_action):
            output_text = parsed_action
        # Otherwise keep the raw output_text (legacy mode or parse failure)

        result = {
            "raw_text": raw_text,
            "output_text": output_text,
            "latency": latency,
            "confidence": confidence,
            "policy_risk": policy_risk,
            "hitl_reason": hitl_reason,
        }
        if isinstance(generated_action_plan, dict):
            merged = {**generated_action_plan, **result}
            # keep the model's own output_text if present, otherwise use raw_text
            merged["output_text"] = output_text or generated_action_plan.get("output_text", "")
            return merged
        return result