"""
dom_state.py – DOM snapshot capture for delta processing.

Captures a hash-based serialised snapshot of a page's interactive DOM so
consecutive states can be compared efficiently without re-sending the full
tree to the LLM each step.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INTERACTIVE_TAGS = frozenset(
    {"a", "button", "input", "select", "textarea", "label", "option",
     "details", "summary"}
)
INTERACTIVE_ROLES = frozenset(
    {"button", "link", "checkbox", "radio", "textbox", "combobox",
     "listbox", "menuitem", "tab", "switch", "searchbox", "spinbutton",
     "slider"}
)

# Attributes we track in a snapshot element
TRACKED_ATTRS = (
    "tag", "text", "aria_label", "placeholder", "elem_id",
    "name", "elem_class", "href", "value", "input_type", "role", "title",
    "disabled", "checked", "selected",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SnapshotElement:
    """Lightweight representation of one interactive DOM node."""
    uid: str                # stable fingerprint hash used as dict key
    tag: str
    text: str
    aria_label: str
    placeholder: str
    elem_id: str
    name: str
    elem_class: str
    href: str
    value: str
    input_type: str
    role: str
    title: str
    disabled: bool
    checked: bool
    selected: bool
    xpath: str              # approximate structural path (for display)

    # ---------- helpers ----------
    def fingerprint(self) -> str:
        """Deterministic hash of all content fields (excludes uid itself)."""
        payload = "|".join(
            str(getattr(self, attr)) for attr in TRACKED_ATTRS
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def to_summary(self) -> str:
        parts = [f"[{self.uid[:6]}] <{self.tag}"]
        if self.elem_id:
            parts.append(f' id="{self.elem_id}"')
        if self.input_type:
            parts.append(f' type="{self.input_type}"')
        if self.aria_label:
            parts.append(f' aria-label="{self.aria_label}"')
        if self.placeholder:
            parts.append(f' placeholder="{self.placeholder}"')
        parts.append(">")
        if self.text:
            parts.append(f" {self.text[:80]}")
        return "".join(parts)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DOMState:
    """Complete snapshot of a page's interactive DOM."""
    url: str
    title: str
    timestamp: float
    # uid → SnapshotElement (ordered by document position)
    elements: Dict[str, SnapshotElement] = field(default_factory=dict)
    # position-ordered uid list (preserves document order)
    order: List[str] = field(default_factory=list)

    # ---------- properties ----------
    @property
    def element_count(self) -> int:
        return len(self.elements)

    def get_element(self, uid: str) -> Optional[SnapshotElement]:
        return self.elements.get(uid)

    # ---------- serialisation ----------
    def to_json(self) -> str:
        d = {
            "url": self.url,
            "title": self.title,
            "timestamp": self.timestamp,
            "order": self.order,
            "elements": {k: v.to_dict() for k, v in self.elements.items()},
        }
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "DOMState":
        d = json.loads(raw)
        elements = {
            k: SnapshotElement(**v) for k, v in d["elements"].items()
        }
        return cls(
            url=d["url"],
            title=d["title"],
            timestamp=d["timestamp"],
            elements=elements,
            order=d["order"],
        )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def _xpath_approx(el) -> str:
    """Build a simple ancestor-chain path (tag[n]) for display purposes."""
    parts: List[str] = []
    for ancestor in reversed(list(el.parents)):
        if ancestor.name:
            siblings = list(ancestor.parent.find_all(ancestor.name, recursive=False)) if ancestor.parent else []
            idx = siblings.index(ancestor) + 1 if ancestor in siblings else 1
            parts.append(f"{ancestor.name}[{idx}]")
    if el.name:
        siblings = list(el.parent.find_all(el.name, recursive=False)) if el.parent else []
        idx = siblings.index(el) + 1 if el in siblings else 1
        parts.append(f"{el.name}[{idx}]")
    return "/" + "/".join(parts) if parts else f"/{el.name or 'unknown'}"


def capture_dom_state(
    html: str,
    url: str = "",
    title: str = "",
    timestamp: float = 0.0,
) -> DOMState:
    """
    Parse *html* and return a DOMState snapshot.

    Parameters
    ----------
    html:      Raw HTML string from the browser.
    url:       Current page URL (for display/tracking).
    title:     Current page title.
    timestamp: Unix epoch at capture time (use time.time()).
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "meta", "noscript", "head"]):
        tag.decompose()

    state = DOMState(url=url, title=title, timestamp=timestamp)

    for el in soup.find_all(True):
        tag_name = (el.name or "").lower()
        role = el.get("role", "").lower()

        is_interactive = (
            tag_name in INTERACTIVE_TAGS
            or role in INTERACTIVE_ROLES
            or el.get("onclick")
            or el.get("tabindex") not in (None, "-1", -1)
        )
        if not is_interactive:
            continue

        style = el.get("style", "").replace(" ", "")
        if "display:none" in style or "visibility:hidden" in style:
            continue

        cls = el.get("class", [])
        cls_str = cls if isinstance(cls, str) else " ".join(cls)

        snap = SnapshotElement(
            uid="",  # filled below
            tag=tag_name,
            text=el.get_text(separator=" ", strip=True)[:200],
            aria_label=el.get("aria-label", ""),
            placeholder=el.get("placeholder", ""),
            elem_id=el.get("id", ""),
            name=el.get("name", ""),
            elem_class=cls_str,
            href=el.get("href", ""),
            value=el.get("value", ""),
            input_type=el.get("type", ""),
            role=role,
            title=el.get("title", ""),
            disabled=el.has_attr("disabled"),
            checked=el.has_attr("checked"),
            selected=el.has_attr("selected"),
            xpath=_xpath_approx(el),
        )
        # uid = hash of structural identity (id/name/tag/xpath) so that the
        # same logical element gets the same uid across page re-renders.
        identity = f"{tag_name}|{snap.elem_id}|{snap.name}|{snap.xpath}"
        snap.uid = hashlib.md5(identity.encode("utf-8")).hexdigest()[:12]

        # Deduplicate: last wins (covers repeated IDs in badly formed HTML)
        state.elements[snap.uid] = snap
        if snap.uid not in state.order:
            state.order.append(snap.uid)

    return state
