"""
privacy_filters.py – PII detection, element anonymisation, and differential privacy.

Three self-contained components:

1. PIIDetector   – regex + heuristic detection of sensitive data in element text/attrs
2. ElementAnonymizer – masks/replaces sensitive content while preserving structure
3. DPScorer      – adds calibrated Laplace noise to element relevance scores
                   (differential privacy for the candidate ranking step)

All classes are stateless and safe to share across threads.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# 1. PIIDetector
# ---------------------------------------------------------------------------
# Pattern registry: name → (compiled regex, label)
_PII_PATTERNS: Dict[str, re.Pattern] = {
    "email":       re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ),
    "phone_us":    re.compile(
        r"\b(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}\b"
    ),
    "ssn":         re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    ),
    "credit_card": re.compile(
        r"\b(?:\d[ -]?){13,16}\b"
    ),
    "ipv4":        re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ),
    "date_of_birth": re.compile(
        r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b"
    ),
    "passport":    re.compile(
        r"\b[A-Z]{1,2}\d{6,9}\b"
    ),
    "aadhaar":     re.compile(          # Indian national ID
        r"\b\d{4}[\s-]\d{4}[\s-]\d{4}\b"
    ),
    "pan_india":   re.compile(          # Indian PAN card
        r"\b[A-Z]{5}\d{4}[A-Z]\b"
    ),
    "password_field": re.compile(       # HTML password inputs (by type/name)
        r"(?i)\bpassword\b"
    ),
}

# Attribute names that are inherently sensitive
_SENSITIVE_ATTR_NAMES: Set[str] = {
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "auth", "credentials", "credit_card", "card_number", "cvv", "ccv",
    "ssn", "social_security", "dob", "date_of_birth",
}


@dataclass
class PIIMatch:
    pii_type: str
    matched_text: str
    attribute: str          # which attribute the match was found in


@dataclass
class PIIReport:
    """Aggregated PII findings for a single element."""
    element_uid: str
    matches: List[PIIMatch] = field(default_factory=list)

    @property
    def has_pii(self) -> bool:
        return bool(self.matches)

    @property
    def pii_types(self) -> Set[str]:
        return {m.pii_type for m in self.matches}

    @property
    def sensitivity_score(self) -> float:
        """0-1 heuristic: higher = more sensitive."""
        weights = {
            "ssn": 1.0, "credit_card": 1.0, "passport": 0.9,
            "aadhaar": 0.9, "pan_india": 0.85, "password_field": 0.8,
            "email": 0.6, "phone_us": 0.6, "date_of_birth": 0.5,
            "ipv4": 0.3,
        }
        if not self.matches:
            return 0.0
        return min(1.0, sum(weights.get(m.pii_type, 0.3) for m in self.matches))


class PIIDetector:
    """Scan element attributes for personally identifiable information."""

    # Attributes to scan
    SCAN_ATTRS = (
        "text", "aria_label", "placeholder", "elem_id",
        "name", "value", "title", "href",
    )

    def detect_in_element(self, element) -> PIIReport:
        """
        Run all PII patterns against an element's text/attribute fields.

        *element* can be a SnapshotElement or any object with the attributes
        listed in SCAN_ATTRS (missing attrs are skipped gracefully).
        """
        report = PIIReport(element_uid=getattr(element, "uid", ""))
        uid = getattr(element, "uid", "")

        for attr in self.SCAN_ATTRS:
            raw = getattr(element, attr, "") or ""
            text = str(raw)

            # Sensitive attribute name check (for name/id/placeholder attrs)
            attr_lower = attr.lower()
            if any(s in attr_lower for s in _SENSITIVE_ATTR_NAMES):
                if text.strip():
                    report.matches.append(PIIMatch("sensitive_attr", text[:20], attr))

            # Regex scan
            for pii_type, pattern in _PII_PATTERNS.items():
                for match in pattern.finditer(text):
                    report.matches.append(
                        PIIMatch(pii_type, match.group(), attr)
                    )

        # Also check if element id/name implies sensitivity
        for name_attr in ("elem_id", "name", "placeholder"):
            val = (getattr(element, name_attr, "") or "").lower()
            if any(s in val for s in _SENSITIVE_ATTR_NAMES):
                report.matches.append(
                    PIIMatch("sensitive_field_name", val, name_attr)
                )

        return report

    def scan_elements(self, elements: List) -> Dict[str, PIIReport]:
        """Return {uid: PIIReport} for all elements with PII."""
        reports: Dict[str, PIIReport] = {}
        for el in elements:
            report = self.detect_in_element(el)
            if report.has_pii:
                reports[getattr(el, "uid", "")] = report
        return reports

    def compute_privacy_score(self, elements: List) -> float:
        """
        Global privacy score: fraction of elements that contain PII.
        Higher → more sensitive page.
        """
        if not elements:
            return 0.0
        pii_count = sum(1 for el in elements if self.detect_in_element(el).has_pii)
        return pii_count / len(elements)


# ---------------------------------------------------------------------------
# 2. ElementAnonymizer
# ---------------------------------------------------------------------------

def _hash_value(text: str, length: int = 8) -> str:
    """One-way hash a value to a short hex string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _mask_pii_in_text(text: str, pii_types: Set[str]) -> str:
    """Replace detected PII patterns in *text* with masked placeholders."""
    for pii_type, pattern in _PII_PATTERNS.items():
        if pii_types and pii_type not in pii_types:
            continue
        label = f"[{pii_type.upper()}_MASKED]"
        text = pattern.sub(label, text)
    return text


class ElementAnonymizer:
    """
    Anonymises sensitive element attributes before they are forwarded to
    an LLM, replacing or hashing PII while preserving structural information
    needed for element selection.
    """

    def __init__(self, hash_ids: bool = True, mask_text: bool = True) -> None:
        self.hash_ids = hash_ids
        self.mask_text = mask_text
        self._detector = PIIDetector()

    def anonymize(self, element, report: Optional[PIIReport] = None):
        """
        Return a copy of *element* with sensitive fields masked.

        Works by creating a dict copy and modifying sensitive values.
        The original element is never mutated.
        """
        # Build a mutable dict copy (works for SnapshotElement via dataclasses)
        import dataclasses
        if dataclasses.is_dataclass(element) and not isinstance(element, type):
            d = dataclasses.asdict(element)
        elif hasattr(element, "__dict__"):
            d = dict(vars(element))
        else:
            raise TypeError(f"Cannot anonymize {type(element)}")

        report = report or self._detector.detect_in_element(element)
        pii_types = report.pii_types

        if self.hash_ids:
            if d.get("elem_id"):
                d["elem_id"] = "id_" + _hash_value(d["elem_id"])
            if d.get("name"):
                d["name"] = "name_" + _hash_value(d["name"])

        if self.mask_text:
            for attr in ("text", "aria_label", "placeholder", "value", "title"):
                if d.get(attr):
                    d[attr] = _mask_pii_in_text(str(d[attr]), pii_types)

        # Reconstruct element of the same type if possible
        try:
            return type(element)(**d)
        except Exception:
            return d   # fallback: return dict

    def anonymize_batch(
        self,
        elements: List,
        reports: Optional[Dict[str, PIIReport]] = None,
    ) -> Tuple[List, int]:
        """
        Anonymise a list of elements.

        Returns (anonymized_list, count_anonymized).
        """
        reports = reports or {}
        result = []
        count = 0
        for el in elements:
            uid = getattr(el, "uid", "")
            report = reports.get(uid) or self._detector.detect_in_element(el)
            if report.has_pii:
                result.append(self.anonymize(el, report=report))
                count += 1
            else:
                result.append(el)
        return result, count


# ---------------------------------------------------------------------------
# 3. DPScorer – Differential Privacy for element ranking
# ---------------------------------------------------------------------------

def _laplace_noise(sensitivity: float, epsilon: float) -> float:
    """
    Sample Laplace noise with scale = sensitivity / epsilon.
    Uses the inverse-CDF method for correctness.
    """
    scale = sensitivity / max(epsilon, 1e-9)
    u = random.uniform(-0.5, 0.5)
    return -scale * math.copysign(1, u) * math.log(1 - 2 * abs(u))


class DPScorer:
    """
    Adds calibrated Laplace noise to element relevance scores to prevent
    the LLM from inferring sensitive element identities through score
    ordering when privacy is required.

    The privacy guarantee is (epsilon, 0)-differential privacy on the
    score vector, with *sensitivity* set to the maximum possible score
    difference between any two adjacent score vectors.
    """

    def __init__(self, epsilon: float = 1.0, sensitivity: float = 10.0) -> None:
        """
        Parameters
        ----------
        epsilon:     Privacy budget (lower → more privacy, more noise).
                     Typical values: 0.1 (strong), 1.0 (moderate), 10.0 (weak).
        sensitivity: Global sensitivity of the scoring function
                     (max δ(score) for adjacent inputs).
        """
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self._total_budget_used: float = 0.0

    @property
    def budget_used(self) -> float:
        return self._total_budget_used

    def add_noise(self, scores: List[float]) -> List[float]:
        """
        Return a noisy copy of *scores*.

        Each call consumes `epsilon` from the privacy budget (basic
        composition).  Track cumulative budget with `budget_used`.
        """
        self._total_budget_used += self.epsilon
        return [
            max(0.0, s + _laplace_noise(self.sensitivity, self.epsilon))
            for s in scores
        ]

    def privatise_ranking(
        self,
        scored_elements: List[Tuple],   # List of (element, score)
        top_n: int = 20,
    ) -> List:
        """
        Apply DP noise to a ranked list and return the top_n elements.

        Parameters
        ----------
        scored_elements: List of (element, score) pairs, highest score first.
        top_n:           How many candidates to keep after noisy re-ranking.

        Returns a list of elements (no scores) in the new noisy order.
        """
        if not scored_elements:
            return []

        elements, scores = zip(*scored_elements)
        noisy_scores = self.add_noise(list(scores))
        reranked = sorted(
            zip(elements, noisy_scores),
            key=lambda x: -x[1],
        )
        return [el for el, _ in reranked[:top_n]]

    def reset_budget(self) -> None:
        self._total_budget_used = 0.0


# ---------------------------------------------------------------------------
# Convenience: PrivacyConfig – bundles all three components
# ---------------------------------------------------------------------------
@dataclass
class PrivacyConfig:
    """Runtime configuration for the full privacy pipeline."""
    enable_pii_detection: bool = True
    enable_anonymization: bool = True
    enable_dp_scoring: bool = False         # off by default (changes rankings)
    dp_epsilon: float = 1.0
    dp_sensitivity: float = 10.0
    hash_ids: bool = True
    mask_text: bool = True

    def build(self) -> "PrivacyPipeline":
        return PrivacyPipeline(self)


class PrivacyPipeline:
    """
    Unified privacy pipeline: detect → anonymise → (optional) DP-noise.

    Use this as the single integration point in run_prune4web.py.
    """

    def __init__(self, config: PrivacyConfig) -> None:
        self.config = config
        self.detector = PIIDetector() if config.enable_pii_detection else None
        self.anonymizer = ElementAnonymizer(
            hash_ids=config.hash_ids,
            mask_text=config.mask_text,
        ) if config.enable_anonymization else None
        self.dp_scorer = DPScorer(
            epsilon=config.dp_epsilon,
            sensitivity=config.dp_sensitivity,
        ) if config.enable_dp_scoring else None
        self._pii_reports: Dict[str, PIIReport] = {}

    def process_elements(
        self,
        elements: List,
        scored_pairs: Optional[List[Tuple]] = None,  # (element, score)
        top_n: int = 20,
    ) -> Tuple[List, Dict[str, PIIReport]]:
        """
        Run the full privacy pipeline on *elements*.

        1. Detect PII across all elements.
        2. Anonymise elements that contain PII.
        3. Optionally apply DP noise to the scored ranking.

        Parameters
        ----------
        elements:     Full candidate element list.
        scored_pairs: Optional (element, score) pairs for DP re-ranking.
                      If None, elements are returned in original order.
        top_n:        Candidates to keep after DP re-ranking.

        Returns (processed_elements, pii_reports).
        """
        # Step 1 – PII detection
        reports: Dict[str, PIIReport] = {}
        if self.detector:
            reports = self.detector.scan_elements(elements)
            self._pii_reports = reports

        # Step 2 – Anonymisation
        if self.anonymizer and reports:
            elements, n_anon = self.anonymizer.anonymize_batch(elements, reports)
            if n_anon:
                print(f"  [privacy] anonymised {n_anon}/{len(elements)} elements")

        # Step 3 – DP scoring
        if self.dp_scorer and scored_pairs:
            elements = self.dp_scorer.privatise_ranking(scored_pairs, top_n=top_n)
        elif scored_pairs:
            elements = [el for el, _ in sorted(scored_pairs, key=lambda x: -x[1])[:top_n]]

        return elements, reports

    @property
    def last_pii_reports(self) -> Dict[str, PIIReport]:
        return self._pii_reports

    def privacy_summary(self, elements: List) -> str:
        if not self.detector:
            return "PII detection disabled."
        score = self.detector.compute_privacy_score(elements)
        pii_count = sum(1 for r in self._pii_reports.values() if r.has_pii)
        return (
            f"Privacy score: {score:.2%}  |  "
            f"PII-containing elements: {pii_count}/{len(elements)}  |  "
            f"DP budget used: {self.dp_scorer.budget_used if self.dp_scorer else 0:.2f}"
        )
