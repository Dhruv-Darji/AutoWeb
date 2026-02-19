"""
HITL Confidence Gate — Composite Confidence Scoring + Human-in-the-Loop Triggering

Computes a weighted composite confidence score from three signals:
    1. LLM self-reported confidence  (weight: 0.5)
    2. Grounding score gap           (weight: 0.3)
    3. Policy risk score             (weight: 0.2)

When the composite score falls below a configurable threshold the gate
flags the step for Human-in-the-Loop (HITL) review.

Usage:
    gate = HITLConfidenceGate(threshold=75)
    composite = gate.compute_composite_confidence(
        llm_confidence=85,
        grounding_top1_score=0.82,
        grounding_top2_score=0.45,
        action_text="[button] Delete -> CLICK",
        policy_risk_keywords=["delete", "remove", "payment"],
    )
    if composite["hitl_triggered"]:
        logger.warning("HITL required!")
"""

import re
from typing import Dict, List

from AutoWeb.src.logger import logger


class HITLConfidenceGate:
    """
    Composite confidence scorer and HITL trigger.

    Formula:
        final = W_LLM * llm_conf + W_POLICY * policy_risk_score

    Default weights: 0.7 / 0.3 (tunable).
    Runs *before* Action Grounding so grounding cost is saved on low-confidence steps.
    """

    # Class-level defaults (can be overridden per instance)
    W_LLM: float = 0.7
    W_POLICY: float = 0.3

    def __init__(
        self,
        threshold: int = 75,
        w_llm: float = None,
        w_policy: float = None,
    ):
        """
        Args:
            threshold:   Composite score below this value triggers HITL (0–100).
            w_llm:       Weight for LLM self-confidence component.
            w_policy:    Weight for policy risk component.
        """
        self.threshold = max(0, min(100, threshold))
        if w_llm is not None:
            self.W_LLM = w_llm
        if w_policy is not None:
            self.W_POLICY = w_policy

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def compute_composite_confidence(
        self,
        llm_confidence: float,
        action_text: str,
        policy_risk_keywords: List[str],
    ) -> Dict:
        """
        Compute the composite HITL confidence score.

        This runs *before* Action Grounding, so only LLM confidence and
        policy-risk signals are used (no grounding scores available yet).

        Args:
            llm_confidence:        Model self-reported confidence (0–100).
            action_text:           Generated action plan text (used for keyword matching).
            policy_risk_keywords:  Risk keywords from PolicyHub.

        Returns:
            Dict with:
                final_confidence   (float 0–100)
                llm_confidence     (float 0–100)
                policy_risk_score  (float 0–100)
                hitl_triggered     (bool)
                threshold          (int)
                matched_keywords   (List[str])
        """
        # --- 1. Clamp LLM confidence ---
        llm_conf = max(0.0, min(100.0, float(llm_confidence)))

        # --- 2. Policy risk score ---
        policy_risk_score, matched_kw = self._compute_policy_risk(
            action_text, policy_risk_keywords
        )

        # --- 3. Weighted composite ---
        final = (
            self.W_LLM * llm_conf
            + self.W_POLICY * policy_risk_score
        )
        final = round(final, 2)

        triggered = final < self.threshold

        result = {
            "final_confidence": final,
            "llm_confidence": llm_conf,
            "policy_risk_score": policy_risk_score,
            "hitl_triggered": triggered,
            "threshold": self.threshold,
            "matched_keywords": matched_kw,
        }

        if triggered:
            logger.debug(
                f"[HITL] Triggered — final={final:.1f} < threshold={self.threshold} "
                f"(llm={llm_conf:.0f}, risk={policy_risk_score:.0f}, "
                f"keywords={matched_kw})"
            )

        return result

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def should_trigger_hitl(self, composite: Dict) -> bool:
        """Check whether HITL is flagged in a composite result dict."""
        return composite.get("hitl_triggered", False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_policy_risk(
        action_text: str, keywords: List[str]
    ) -> tuple:
        """
        Score how risky the action text is with respect to policy keywords.

        Returns:
            (score, matched_keywords)

            score mapping:
                100  — no policy-risk keywords found (safe)
                 50  — 1 keyword found (moderate risk)
                 20  — 2+ keywords found (high risk)
        """
        if not action_text or not keywords:
            return 100.0, []

        text_lower = action_text.lower()
        matched = [kw for kw in keywords if kw.lower() in text_lower]

        if len(matched) == 0:
            return 100.0, []
        elif len(matched) == 1:
            return 50.0, matched
        else:
            return 20.0, matched
