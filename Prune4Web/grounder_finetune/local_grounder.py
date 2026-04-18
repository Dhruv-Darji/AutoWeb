"""
local_grounder.py — Drop-in replacement for Prune4Web's action_grounder().

Loads the LoRA-fine-tuned Qwen2.5-0.5B grounder (trained by 02_finetune.py) and
exposes a `ground()` function with the same signature as `action_grounder()`
from run_prune4web.py, so the live pipeline can swap out the OpenAI grounder
for a fully local one:

    from Prune4Web.grounder_finetune.local_grounder import ground as action_grounder

Design
------
* Lazy, process-wide singleton: the base model + adapter load only on the first
  call. Subsequent calls reuse the in-memory model.
* Base model is quantised to 4-bit (nf4) via bitsandbytes to fit in ~1 GB VRAM.
* Inference uses `model.generate()` with do_sample=False (deterministic).
* The prompt format is byte-for-byte identical to what 01_generate_data.py wrote
  into the training JSONL, so the model sees exactly the same layout at
  inference as it did during training.
* Output is parsed with the same robust JSON extractor used by 03_evaluate.py.

Environment variables
---------------------
  GROUNDER_BASE_MODEL   base (non-finetuned) Qwen2.5-0.5B directory
  GROUNDER_OUTPUT_MODEL LoRA adapter directory (produced by 02_finetune.py)
  GROUNDER_USE_LOCAL    if set to "1", run_prune4web.py can check this and
                        import this module instead of the OpenAI grounder

Usage from run_prune4web.py
---------------------------
    import os
    if os.getenv("GROUNDER_USE_LOCAL") == "1":
        from Prune4Web.grounder_finetune.local_grounder import ground as action_grounder
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from dotenv import load_dotenv

# -------------------- Bootstrap --------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")
sys.path.insert(0, str(_PROJECT_ROOT))

# Reuse the exact system prompt used at training time
from Prune4Web.run_prune4web import GROUNDER_SYSTEM  # noqa: E402
from Prune4Web.run_prune4web import ElementNode  # noqa: E402


BASE_MODEL_DIR = Path(os.getenv("GROUNDER_BASE_MODEL",
                                "D:/Environments/Models/Qwen2.5-0.5B-Instruct"))
ADAPTER_DIR = Path(os.getenv("GROUNDER_OUTPUT_MODEL",
                             "D:/Environments/Models/Qwen2.5-0.5B-Prune4Web-Grounder"))

_MAX_NEW_TOKENS = int(os.getenv("GROUNDER_MAX_NEW_TOKENS", "128"))
_MAX_INPUT_TOKENS = int(os.getenv("GROUNDER_MAX_INPUT_TOKENS", "1536"))


# -------------------- Singleton state --------------------
_tokenizer = None
_model = None
_device = None


def _load_model():
    """Load base model in 4-bit + LoRA adapter. Called once per process."""
    global _tokenizer, _model, _device

    if _model is not None:
        return

    # Imports kept inside the function so simply importing this module is cheap.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not BASE_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Base model not found at {BASE_MODEL_DIR}. Set GROUNDER_BASE_MODEL."
        )
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"Fine-tuned adapter not found at {ADAPTER_DIR}. Run 02_finetune.py first."
        )

    print(f"[local_grounder] Loading base model from {BASE_MODEL_DIR} (4-bit)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_DIR),
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    print(f"[local_grounder] Attaching LoRA adapter from {ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    model.eval()

    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"[local_grounder] Model ready. VRAM: {vram:.2f} GB")

    _tokenizer = tokenizer
    _model = model
    _device = model.device


# -------------------- Output parsing --------------------
def _extract_json(text: str) -> Optional[Dict]:
    """Pull the first well-formed JSON object out of the model output."""
    if not text:
        return None
    match = re.search(r"\{[^{}]*?\"element_uid\"[^{}]*\}", text, re.DOTALL)
    if not match:
        match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    raw = match.group(0)
    try:
        return json.loads(raw)
    except Exception:
        try:
            return json.loads(raw.rsplit("}", 1)[0] + "}")
        except Exception:
            return None


def _coerce_result(parsed: Optional[Dict], num_candidates: int) -> Dict:
    """Match the exact dict schema expected by run_prune4web.py's caller."""
    if not isinstance(parsed, dict):
        return {
            "element_uid": -1,
            "action": "click",
            "value": "",
            "confidence": 0.0,
            "reasoning": "local grounder: failed to parse JSON from model output",
        }
    try:
        uid = int(parsed.get("element_uid", -1))
    except Exception:
        uid = -1
    # element_uid is in the range [0, num_candidates-1]; anything outside -> -1
    if uid < 0 or uid >= num_candidates:
        uid = -1

    action = str(parsed.get("action", "click")).strip().lower() or "click"
    value = str(parsed.get("value", "") or "")
    try:
        confidence = float(parsed.get("confidence", 0.9))
    except Exception:
        confidence = 0.9
    reasoning = str(parsed.get("reasoning", ""))[:240]

    return {
        "element_uid": uid,
        "action": action,
        "value": value,
        "confidence": confidence,
        "reasoning": reasoning,
    }


# -------------------- Public API --------------------
def ground(sub_task: str, candidates: List[ElementNode], action_value: str = "") -> Dict:
    """
    Local-model equivalent of run_prune4web.action_grounder().

    Parameters mirror action_grounder() exactly:
      sub_task     : natural-language sub-task from the planner
      candidates   : top-N ElementNode candidates from programmatic_element_filter
      action_value : optional value hint (typing text, select value...)

    Returns a dict with keys: element_uid, action, value, confidence, reasoning.
    """
    _load_model()

    # The training data used contiguous uids 0..N-1 (build_candidate_window
    # re-indexes after shuffling).  At live inference the candidates carry
    # their original DOM uids (e.g. 42, 197, 350).  We temporarily re-map
    # to 0..N-1 so the model sees the same layout it was trained on, then
    # translate the predicted index back to the real uid.
    original_uids = [el.uid for el in candidates]
    for idx, el in enumerate(candidates):
        el.uid = idx

    candidate_lines = [f"  {i + 1}. {el.to_summary()}" for i, el in enumerate(candidates)]
    user_msg = (
        f"Sub-task: {sub_task}\nValue hint: {action_value}\n\n"
        f"Candidate elements (ranked by relevance):\n" + "\n".join(candidate_lines)
    )

    messages = [
        {"role": "system", "content": GROUNDER_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if os.getenv("GROUNDER_DISABLE_THINKING", "0") == "1":
        chat_kwargs["enable_thinking"] = False
    try:
        prompt_text = _tokenizer.apply_chat_template(messages, **chat_kwargs)
    except TypeError:
        chat_kwargs.pop("enable_thinking", None)
        prompt_text = _tokenizer.apply_chat_template(messages, **chat_kwargs)
    inputs = _tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=_MAX_INPUT_TOKENS,
    ).to(_device)

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=_tokenizer.pad_token_id,
        )

    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    gen_text = _tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Restore original uids on the candidate objects
    for el, orig_uid in zip(candidates, original_uids):
        el.uid = orig_uid

    parsed = _extract_json(gen_text)
    result = _coerce_result(parsed, num_candidates=len(candidates))

    # Map predicted 0-based index back to the real DOM uid
    pred_idx = result["element_uid"]
    if 0 <= pred_idx < len(original_uids):
        result["element_uid"] = original_uids[pred_idx]

    return result


# -------------------- Smoke test --------------------
if __name__ == "__main__":
    """Quick sanity check: build a fake candidate list and run the grounder."""
    from dataclasses import dataclass

    @dataclass
    class FakeEl:
        uid: int
        text: str
        tag: str = "button"

        def to_summary(self) -> str:
            return (
                f"[uid={self.uid}] tag={self.tag} text='{self.text}' "
                f"aria_label='{self.text}'"
            )

    fake_candidates = [
        FakeEl(uid=0, text="Search"),
        FakeEl(uid=1, text="Add to Bag"),
        FakeEl(uid=2, text="Sign in"),
        FakeEl(uid=3, text="Home"),
        FakeEl(uid=4, text="Cart"),
    ]
    result = ground(
        sub_task="Click the Add to Bag button to buy the product",
        candidates=fake_candidates,
        action_value="",
    )
    print("\nGrounder result:")
    print(json.dumps(result, indent=2))
    expected_uid = 1
    if result["element_uid"] == expected_uid:
        print(f"[OK] Correct element (uid={expected_uid})")
    else:
        print(f"[X]  Expected uid={expected_uid}, got uid={result['element_uid']}")
