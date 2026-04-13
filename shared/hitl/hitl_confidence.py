"""Shared HITL confidence gate used by AutoWeb and Prune4Web."""

from __future__ import annotations

from typing import Dict, List


class HITLConfidenceGate:
    """Risk-based human-in-the-loop trigger."""

    def __init__(self, low_confidence_floor: int = 30, threshold: int = 75):
        self.low_confidence_floor = max(0, min(100, int(low_confidence_floor)))
        # Stored for backward compatibility with existing callers.
        self.threshold = int(threshold)

    def evaluate(
        self,
        llm_confidence: float,
        policy_risk_flag: bool,
        action_text: str,
        policy_risk_keywords: List[str],
        hitl_reason: str = "",
    ) -> Dict:
        """Return HITL decision and trigger metadata."""
        llm_conf = max(0.0, min(100.0, float(llm_confidence)))
        combined_text = f"{action_text} {hitl_reason}".strip()
        matched_kw = self._scan_keywords(combined_text, policy_risk_keywords)

        trigger_reasons: List[str] = []
        if bool(policy_risk_flag):
            trigger_reasons.append("llm_flagged_policy_risk")
        if matched_kw:
            trigger_reasons.append(f"risk_keywords_matched: {matched_kw}")
        if llm_conf < self.low_confidence_floor:
            trigger_reasons.append(f"very_low_confidence ({llm_conf:.0f} < {self.low_confidence_floor})")

        triggered = len(trigger_reasons) > 0
        return {
            "hitl_triggered": triggered,
            "trigger_reasons": trigger_reasons,
            "llm_confidence": llm_conf,
            "policy_risk_flag": bool(policy_risk_flag),
            "matched_keywords": matched_kw,
            "low_confidence_floor": self.low_confidence_floor,
        }

    def compute_composite_confidence(
        self,
        llm_confidence: float,
        action_text: str,
        policy_risk_keywords: List[str],
        policy_risk_flag: bool = False,
        hitl_reason: str = "",
    ) -> Dict:
        """Backward-compatible wrapper around evaluate()."""
        result = self.evaluate(
            llm_confidence=llm_confidence,
            policy_risk_flag=policy_risk_flag,
            action_text=action_text,
            policy_risk_keywords=policy_risk_keywords,
            hitl_reason=hitl_reason,
        )
        result["final_confidence"] = result["llm_confidence"]
        result["policy_risk_score"] = 20.0 if result["matched_keywords"] else 100.0
        result["threshold"] = self.threshold
        return result

    def should_trigger_hitl(self, result: Dict) -> bool:
        return bool(result.get("hitl_triggered", False))

    @staticmethod
    def _scan_keywords(action_text: str, keywords: List[str]) -> List[str]:
        if not action_text or not keywords:
            return []
        text_lower = action_text.lower()
        return [kw for kw in keywords if kw and kw.lower() in text_lower]

