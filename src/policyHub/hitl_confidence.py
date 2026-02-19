"""
HITL Confidence Gate — Risk-Based Human-in-the-Loop Triggering

Decides whether a predicted action needs human review based on three
independent risk signals (any one is sufficient to trigger HITL):

    1. **LLM policy_risk flag** — the model explicitly flagged the action
       as involving a destructive, financial, credential, CAPTCHA, or
       account-change operation.
    2. **Keyword scan** — the action text contains one or more risk keywords
       from the PolicyHub policy file (delete, payment, password, …).
    3. **Very low confidence** — the model's self-reported confidence is
       below a configurable floor (default 30), meaning it is essentially
       guessing.

Normal navigation with moderate confidence (e.g., 60 on a dense Wikipedia
page) does **not** trigger HITL — ambiguity ≠ risk.

Usage:
    gate = HITLConfidenceGate()
    result = gate.evaluate(
        llm_confidence=72,
        policy_risk_flag=False,
        action_text="[link] Wikibooks -> CLICK",
        policy_risk_keywords=["delete", "remove", "payment"],
    )
    if result["hitl_triggered"]:
        logger.warning("HITL required!")
"""

import re
from typing import Dict, List

from AutoWeb.src.logger import logger


class HITLConfidenceGate:
    """
    Risk-based HITL trigger.

    Triggers when:
        - LLM flagged ``policy_risk = True``   OR
        - Action text contains policy risk keywords   OR
        - LLM confidence < ``low_confidence_floor``

    Does NOT trigger merely because a page is complex or confidence is
    moderate on a safe action.
    """

    def __init__(
        self,
        low_confidence_floor: int = 30,
        threshold: int = 75,          # kept for backward compat / config, but unused by default
    ):
        """
        Args:
            low_confidence_floor: Confidence below this value triggers HITL
                                  regardless of risk signals (model is guessing).
            threshold:            Legacy composite threshold (retained for config compatibility).
        """
        self.low_confidence_floor = max(0, min(100, low_confidence_floor))
        self.threshold = threshold  # stored but not used in risk-based logic

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        llm_confidence: float,
        policy_risk_flag: bool,
        action_text: str,
        policy_risk_keywords: List[str],
        hitl_reason: str = "",
    ) -> Dict:
        """
        Decide whether the step needs HITL review.

        Args:
            llm_confidence:        Model self-reported confidence (0–100).
            policy_risk_flag:      Model-reported boolean — True if the action
                                   may involve policy-sensitive operations.
            action_text:           Generated action plan text (keyword scan).
            policy_risk_keywords:  Risk keywords from PolicyHub.
            hitl_reason:           Model-provided reason string (also scanned for keywords).

        Returns:
            Dict with:
                hitl_triggered      (bool)
                trigger_reasons     (List[str])  — why HITL was triggered (empty if not)
                llm_confidence      (float 0–100)
                policy_risk_flag    (bool)
                matched_keywords    (List[str])
        """
        llm_conf = max(0.0, min(100.0, float(llm_confidence)))

        # --- 1. Keyword scan (action text + hitl_reason) ---
        combined_text = f"{action_text} {hitl_reason}".strip()
        matched_kw = self._scan_keywords(combined_text, policy_risk_keywords)

        # --- 2. Collect trigger reasons ---
        trigger_reasons: List[str] = []

        if policy_risk_flag:
            trigger_reasons.append("llm_flagged_policy_risk")

        if matched_kw:
            trigger_reasons.append(f"risk_keywords_matched: {matched_kw}")

        if llm_conf < self.low_confidence_floor:
            trigger_reasons.append(f"very_low_confidence ({llm_conf:.0f} < {self.low_confidence_floor})")

        triggered = len(trigger_reasons) > 0

        result = {
            "hitl_triggered": triggered,
            "trigger_reasons": trigger_reasons,
            "llm_confidence": llm_conf,
            "policy_risk_flag": policy_risk_flag,
            "matched_keywords": matched_kw,
            "low_confidence_floor": self.low_confidence_floor,
        }

        if triggered:
            logger.debug(
                f"[HITL] Triggered — reasons={trigger_reasons}  "
                f"(llm_conf={llm_conf:.0f}, risk_flag={policy_risk_flag}, "
                f"keywords={matched_kw})"
            )
        else:
            logger.debug(
                f"[HITL] Not triggered — llm_conf={llm_conf:.0f}, "
                f"risk_flag={policy_risk_flag}, keywords={matched_kw}"
            )

        return result

    # ------------------------------------------------------------------
    # Backward-compatible alias
    # ------------------------------------------------------------------

    def compute_composite_confidence(
        self,
        llm_confidence: float,
        action_text: str,
        policy_risk_keywords: List[str],
        policy_risk_flag: bool = False,
        hitl_reason: str = "",
    ) -> Dict:
        """Backward-compatible wrapper around :meth:`evaluate`.

        Maps the old interface to the new risk-based logic.  The returned
        dict includes ``final_confidence`` and ``threshold`` keys so that
        callers that still read those fields keep working, but the actual
        HITL decision is risk-based.
        """
        result = self.evaluate(
            llm_confidence=llm_confidence,
            policy_risk_flag=policy_risk_flag,
            action_text=action_text,
            policy_risk_keywords=policy_risk_keywords,
            hitl_reason=hitl_reason,
        )
        # Add legacy keys for callers that still reference them
        result["final_confidence"] = result["llm_confidence"]
        result["policy_risk_score"] = 20.0 if result["matched_keywords"] else 100.0
        result["threshold"] = self.threshold
        return result

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def should_trigger_hitl(self, result: Dict) -> bool:
        """Check whether HITL is flagged in a result dict."""
        return result.get("hitl_triggered", False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_keywords(action_text: str, keywords: List[str]) -> List[str]:
        """Return list of policy risk keywords found in the action text."""
        if not action_text or not keywords:
            return []
        text_lower = action_text.lower()
        return [kw for kw in keywords if kw.lower() in text_lower]
