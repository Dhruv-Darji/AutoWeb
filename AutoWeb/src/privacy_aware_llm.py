"""
privacy_aware_llm.py – Privacy-preserving wrapper for LLM calls in Prune4Web.

Intercepts every LLM prompt that touches DOM element content, applies the
PrivacyPipeline (PII masking, element anonymisation), injects a privacy
instruction block into the system prompt, and logs what was masked.

Drop-in replacement for the bare ``llm_call`` function in run_prune4web.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    from AutoWeb.src.privacy_filters import (
        PIIDetector, PIIReport, PrivacyConfig, PrivacyPipeline, _mask_pii_in_text,
    )
except ImportError:
    from src.privacy_filters import (
        PIIDetector, PIIReport, PrivacyConfig, PrivacyPipeline, _mask_pii_in_text,
    )

# ---------------------------------------------------------------------------
# Privacy system-prompt injection
# ---------------------------------------------------------------------------
_PRIVACY_INSTRUCTION = """
[PRIVACY DIRECTIVE]
The following content has been pre-processed to protect user privacy:
  - Personally Identifiable Information (PII) has been replaced with
    placeholder tokens (e.g. [EMAIL_MASKED], [PHONE_US_MASKED]).
  - Element IDs and names may be hashed (e.g. id_3f2a…, name_7c1b…).
  - Original values are NOT recoverable from the placeholders.

Your task is to select the correct interactive element or generate an action
using ONLY the structural and semantic information provided.  Do NOT attempt
to reconstruct, guess, or request the original PII values.
[END PRIVACY DIRECTIVE]
""".strip()

_PRIVACY_REMINDER = (
    "\n\n[Note: Element attributes may contain privacy placeholders. "
    "Treat them as opaque identifiers when selecting elements.]"
)


# ---------------------------------------------------------------------------
# Prompt-level PII scrubber (for ad-hoc text fields)
# ---------------------------------------------------------------------------
_DETECTOR = PIIDetector()


def scrub_prompt_text(text: str) -> str:
    """
    Replace raw PII patterns in free-text prompt content with masked tokens.
    Used as a last-resort safety net on the assembled prompt string.
    """
    from src.privacy_filters import _PII_PATTERNS
    result = text
    for pii_type, pattern in _PII_PATTERNS.items():
        result = pattern.sub(f"[{pii_type.upper()}_MASKED]", result)
    return result


# ---------------------------------------------------------------------------
# PrivacyAwareLLM
# ---------------------------------------------------------------------------
@dataclass
class PrivacyCallLog:
    """Record of one privacy-aware LLM invocation."""
    stage: str
    pii_found: int              # total PII matches across all elements
    elements_masked: int        # how many elements had content changed
    prompt_scrubbed: bool       # whether the final prompt was also scrubbed
    dp_noise_applied: bool


class PrivacyAwareLLM:
    """
    Wraps a raw ``llm_call`` callable with privacy preprocessing.

    Usage
    -----
    # In run_prune4web.py setup:
    from src.privacy_aware_llm import PrivacyAwareLLM, PrivacyConfig

    privacy_cfg = PrivacyConfig(
        enable_pii_detection=True,
        enable_anonymization=True,
        enable_dp_scoring=False,
    )
    privacy_llm = PrivacyAwareLLM(raw_llm_call=llm_call, config=privacy_cfg)

    # Replace bare llm_call with:
    result = privacy_llm.call(messages, stage="planner", ...)
    """

    def __init__(
        self,
        raw_llm_call: Callable[..., str],
        config: Optional[PrivacyConfig] = None,
        scrub_final_prompt: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        raw_llm_call:        The original llm_call function from run_prune4web.py.
        config:              PrivacyConfig; defaults to detection + anonymisation.
        scrub_final_prompt:  Apply regex PII scrubbing to the fully-assembled
                             message string as a final safety pass.
        """
        self.raw_call = raw_llm_call
        self.config = config or PrivacyConfig()
        self.pipeline = PrivacyPipeline(self.config)
        self.scrub_final = scrub_final_prompt
        self.call_log: List[PrivacyCallLog] = []

    # ------------------------------------------------------------------
    # Core call method
    # ------------------------------------------------------------------
    def call(
        self,
        messages: List[Dict[str, Any]],
        stage: str,
        elements: Optional[List] = None,
        **kwargs,
    ) -> str:
        """
        Privacy-aware replacement for ``llm_call(messages, stage, …)``.

        Parameters
        ----------
        messages:  The prompt messages list (will be deep-copied and modified).
        stage:     Pipeline stage label ("planner", "filter", "grounder").
        elements:  If provided, these are scanned for PII and the pipeline
                   can anonymise their summaries in the user message.
        **kwargs:  Passed verbatim to the underlying raw_llm_call.
        """
        # Deep-copy messages so we never mutate the caller's list
        import copy
        safe_messages = copy.deepcopy(messages)

        # --- Step 1: Inject privacy instruction into system message ----------
        safe_messages = _inject_privacy_instruction(safe_messages)

        # --- Step 2: Scan / anonymise elements -------------------------------
        n_pii = 0
        n_masked = 0
        if elements:
            reports = self.pipeline.detector.scan_elements(elements) if self.pipeline.detector else {}
            n_pii = sum(len(r.matches) for r in reports.values())
            if reports and self.pipeline.anonymizer:
                _, n_masked = self.pipeline.anonymizer.anonymize_batch(elements, reports)
                # Rebuild user message with anonymised element summaries
                safe_messages = _replace_element_summaries(safe_messages, elements, reports)

        # --- Step 3: Final prompt scrub (regex sweep) -----------------------
        scrubbed = False
        if self.scrub_final:
            safe_messages, scrubbed = _scrub_messages(safe_messages)

        # --- Step 4: Log ----------------------------------------------------
        self.call_log.append(PrivacyCallLog(
            stage=stage,
            pii_found=n_pii,
            elements_masked=n_masked,
            prompt_scrubbed=scrubbed,
            dp_noise_applied=self.config.enable_dp_scoring,
        ))

        if n_pii or scrubbed:
            print(
                f"  [privacy:{stage}] pii_matches={n_pii} "
                f"elements_masked={n_masked} scrubbed={scrubbed}"
            )

        # --- Step 5: Forward to real LLM ------------------------------------
        return self.raw_call(messages=safe_messages, stage=stage, **kwargs)

    # ------------------------------------------------------------------
    # Convenience: privacy stats
    # ------------------------------------------------------------------
    def print_privacy_summary(self) -> None:
        print("\n" + "=" * 72)
        print("PRIVACY CALL SUMMARY")
        print("=" * 72)
        total_pii = sum(c.pii_found for c in self.call_log)
        total_masked = sum(c.elements_masked for c in self.call_log)
        scrubbed_calls = sum(1 for c in self.call_log if c.prompt_scrubbed)
        for i, c in enumerate(self.call_log, 1):
            print(
                f"{i:02d}. stage={c.stage:<8}  pii={c.pii_found:<4} "
                f"masked={c.elements_masked:<4} scrubbed={c.prompt_scrubbed}"
            )
        print("-" * 72)
        print(f"TOTAL PII matches found : {total_pii}")
        print(f"TOTAL elements masked   : {total_masked}")
        print(f"Calls with scrubbing    : {scrubbed_calls}/{len(self.call_log)}")
        print("=" * 72)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _inject_privacy_instruction(messages: List[Dict]) -> List[Dict]:
    """Prepend the privacy directive to the system message content."""
    for msg in messages:
        if msg.get("role") == "system":
            original = msg.get("content", "")
            if isinstance(original, str):
                msg["content"] = _PRIVACY_INSTRUCTION + "\n\n" + original
            elif isinstance(original, list):
                # Multi-modal content list
                msg["content"] = [{"type": "text", "text": _PRIVACY_INSTRUCTION}] + original
            return messages  # only patch the first system message

    # No system message found – prepend one
    messages.insert(0, {"role": "system", "content": _PRIVACY_INSTRUCTION})
    return messages


def _replace_element_summaries(
    messages: List[Dict],
    elements: List,
    reports: Dict[str, PIIReport],
) -> List[Dict]:
    """
    If any element with PII appears as a summary string in the user message,
    replace it with the anonymised version.

    This is a best-effort text replacement: it looks for the uid prefix
    pattern ``[uid6]`` in the user content and substitutes masked summaries.
    """
    # Build uid → masked_summary map
    masked_map: Dict[str, str] = {}
    for el in elements:
        uid = getattr(el, "uid", "")
        if uid in reports:
            # Build a "masked" summary by replacing PII in text fields
            pii_types = reports[uid].pii_types
            original_summary = el.to_summary() if hasattr(el, "to_summary") else str(el)
            masked = _mask_pii_in_text(original_summary, pii_types)
            masked_map[uid[:6]] = masked

    if not masked_map:
        return messages

    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            for short_uid, masked_summary in masked_map.items():
                # Pattern: "[uid6] <tag ..." appearing in the message
                content = re.sub(
                    rf"\[{re.escape(short_uid)}\][^\n]*",
                    masked_summary,
                    content,
                )
            msg["content"] = content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    for short_uid, masked_summary in masked_map.items():
                        text = re.sub(
                            rf"\[{re.escape(short_uid)}\][^\n]*",
                            masked_summary,
                            text,
                        )
                    part["text"] = text

    return messages


def _scrub_messages(messages: List[Dict]) -> tuple[List[Dict], bool]:
    """Apply regex PII scrubbing to all text content in messages."""
    changed = False
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            new_content = scrub_prompt_text(content)
            if new_content != content:
                msg["content"] = new_content
                changed = True
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    original = part.get("text", "")
                    new_text = scrub_prompt_text(original)
                    if new_text != original:
                        part["text"] = new_text
                        changed = True
    return messages, changed


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------
def build_privacy_aware_llm(
    raw_llm_call: Callable[..., str],
    enable_dp: bool = False,
    dp_epsilon: float = 1.0,
) -> PrivacyAwareLLM:
    """
    Convenience factory used in run_prune4web.py to set up privacy with
    sensible defaults.

    Parameters
    ----------
    raw_llm_call: The original llm_call from run_prune4web.py.
    enable_dp:    Enable differential privacy on element scoring.
    dp_epsilon:   DP budget (lower = more privacy noise).
    """
    cfg = PrivacyConfig(
        enable_pii_detection=True,
        enable_anonymization=True,
        enable_dp_scoring=enable_dp,
        dp_epsilon=dp_epsilon,
    )
    return PrivacyAwareLLM(raw_llm_call=raw_llm_call, config=cfg, scrub_final_prompt=True)
