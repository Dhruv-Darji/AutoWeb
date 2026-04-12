"""
PolicyHub — Global + Site-Specific Policy Management

Stores and serves policy constraints that are injected into Action Generation
prompts. Policies guide the LLM to avoid risky/destructive actions and to
self-assess confidence when actions may violate business rules.

Usage:
    hub = PolicyHub()                         # loads from default JSON
    hub = PolicyHub("path/to/policies.json")  # loads from custom path

    policy_text = hub.get_active_policy("amazon.com")
    keywords    = hub.get_policy_risk_keywords()
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from AutoWeb.src.logger import logger


# ---------------------------------------------------------------------------
# Hardcoded defaults (used when JSON file is missing or incomplete)
# ---------------------------------------------------------------------------

DEFAULT_POLICIES: List[str] = [
    "Do not submit payment forms automatically.",
    "Do not click destructive actions like delete or remove without confirmation.",
    "Do not change account passwords.",
    "Do not share sensitive personal information.",
    "Do not approve financial transactions without explicit user confirmation.",
]

# Keywords extracted from default policies — used for policy-risk scoring
_DEFAULT_RISK_KEYWORDS: List[str] = [
    "delete", "remove", "purchase", "payment", "password",
    "approve", "submit", "transfer", "checkout", "cancel",
    "terminate", "destroy", "discard", "unsubscribe",
]


class PolicyHub:
    """
    Manages global default policies and per-site policy overrides.

    Policies are simple string rules (natural-language bullet points) that
    get injected into the Action Generation prompt.  At runtime the active
    policy is the union of ``default_policy`` and any ``site_specific``
    rules matching the current domain.
    """

    def __init__(self, json_path: Optional[str] = None):
        """
        Args:
            json_path: Path to a ``policies.json`` file.  If *None* the
                       default file ``src/policyHub/policies.json`` next to
                       this module is used.  If the file doesn't exist the
                       hardcoded ``DEFAULT_POLICIES`` are used.
        """
        self._json_path: str = json_path or str(
            Path(__file__).parent / "policies.json"
        )
        self.default_policy: List[str] = list(DEFAULT_POLICIES)
        self.site_specific: Dict[str, List[str]] = {}
        self._risk_keywords: List[str] = list(_DEFAULT_RISK_KEYWORDS)

        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load policies from JSON, falling back to hardcoded defaults."""
        if not os.path.isfile(self._json_path):
            logger.info(
                f"[PolicyHub] No policies file at {self._json_path} — using defaults"
            )
            return

        try:
            with open(self._json_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            if isinstance(data.get("default_policy"), list):
                self.default_policy = [str(r) for r in data["default_policy"]]

            if isinstance(data.get("site_specific"), dict):
                self.site_specific = {
                    str(k): [str(r) for r in v]
                    for k, v in data["site_specific"].items()
                    if isinstance(v, list)
                }

            # Merge any extra risk keywords from JSON
            if isinstance(data.get("risk_keywords"), list):
                extra = [str(k).lower() for k in data["risk_keywords"]]
                combined = set(self._risk_keywords) | set(extra)
                self._risk_keywords = sorted(combined)

            logger.info(
                f"[PolicyHub] Loaded {len(self.default_policy)} default + "
                f"{len(self.site_specific)} site-specific policy sets "
                f"from {self._json_path}"
            )
        except Exception as exc:
            logger.warning(
                f"[PolicyHub] Failed to load {self._json_path}: {exc} — using defaults"
            )

    def _save(self) -> None:
        """Persist current policies back to JSON."""
        data = {
            "default_policy": self.default_policy,
            "site_specific": self.site_specific,
            "risk_keywords": self._risk_keywords,
        }
        try:
            os.makedirs(os.path.dirname(self._json_path) or ".", exist_ok=True)
            with open(self._json_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            logger.debug(f"[PolicyHub] Saved policies to {self._json_path}")
        except Exception as exc:
            logger.error(f"[PolicyHub] Failed to save policies: {exc}")

    # ------------------------------------------------------------------
    # Public API — Query
    # ------------------------------------------------------------------

    def get_active_policy(self, domain: Optional[str] = None) -> str:
        """
        Return the merged policy text for a given domain.

        Combines ``default_policy`` with any matching ``site_specific``
        rules, formatted as a bullet-point list suitable for prompt
        injection.

        Args:
            domain: Website domain (e.g. ``"amazon.com"``).  If *None* only
                    default policies are returned.

        Returns:
            Multiline string with one bullet per rule, or empty string if
            no policies are configured.
        """
        rules: List[str] = list(self.default_policy)

        if domain:
            # Try exact match first, then substring match
            domain_lower = domain.lower().strip()
            for key, site_rules in self.site_specific.items():
                if key.lower().strip() == domain_lower or domain_lower in key.lower():
                    rules.extend(site_rules)
                    break

        if not rules:
            return ""

        return "\n".join(f"• {rule}" for rule in rules)

    def get_policy_risk_keywords(self) -> List[str]:
        """
        Return risk keywords derived from all policies.

        These are single words (lower-case) that indicate potentially
        dangerous or policy-violating actions.  Used by the HITL
        confidence gate to compute the policy-risk component.
        """
        return list(self._risk_keywords)

    def get_stats(self) -> Dict:
        """Summary statistics for logging."""
        return {
            "default_rules": len(self.default_policy),
            "site_specific_domains": len(self.site_specific),
            "site_specific_rules": sum(
                len(v) for v in self.site_specific.values()
            ),
            "risk_keywords": len(self._risk_keywords),
        }

    # ------------------------------------------------------------------
    # Public API — Mutation
    # ------------------------------------------------------------------

    def add_site_policy(
        self, domain: str, rules: List[str], *, persist: bool = True
    ) -> None:
        """
        Add or extend site-specific policy rules for *domain*.

        Args:
            domain:  Website domain key (e.g. ``"amazon.com"``).
            rules:   List of policy rule strings to add.
            persist: If *True*, save updated policies to JSON.
        """
        existing = self.site_specific.get(domain, [])
        existing.extend(rules)
        self.site_specific[domain] = existing
        logger.info(
            f"[PolicyHub] Added {len(rules)} rules for '{domain}' "
            f"(total: {len(existing)})"
        )
        if persist:
            self._save()

    def update_default_policy(
        self, rules: List[str], *, replace: bool = False, persist: bool = True
    ) -> None:
        """
        Update the global default policy.

        Args:
            rules:   New rule strings.
            replace: If *True*, replace existing defaults.  Otherwise append.
            persist: If *True*, save to JSON.
        """
        if replace:
            self.default_policy = list(rules)
        else:
            self.default_policy.extend(rules)
        logger.info(
            f"[PolicyHub] Default policy updated ({len(self.default_policy)} rules)"
        )
        if persist:
            self._save()

    def add_risk_keywords(
        self, keywords: List[str], *, persist: bool = True
    ) -> None:
        """Add extra risk keywords."""
        combined = set(self._risk_keywords) | {k.lower() for k in keywords}
        self._risk_keywords = sorted(combined)
        if persist:
            self._save()
