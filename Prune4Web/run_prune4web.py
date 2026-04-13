"""Interactive Prune4Web runner with live browser preview and usage accounting.

Enhanced with:
  - DOM Delta Processing (src/dom_diff.py) – only changed/new elements sent to LLM
  - Privacy-Aware LLM (src/privacy_aware_llm.py) – PII masking before LLM calls
"""

from __future__ import annotations

import asyncio
import base64
import getpass
import json
import os
import re
import sys
import tempfile
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from openai import BadRequestError, OpenAI
from playwright.async_api import Page, async_playwright
from rapidfuzz import fuzz
from dotenv import load_dotenv

# Load .env from project root before anything reads os.getenv()
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
load_dotenv(Path(_PROJECT_ROOT) / ".env")

# --- New research contributions -------------------------------------------
# Add project root to sys.path so AutoWeb/src imports work
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from AutoWeb.src.dom_diff import DOMDeltaProcessor
    from AutoWeb.src.privacy_aware_llm import build_privacy_aware_llm, PrivacyAwareLLM
    from AutoWeb.src.privacy_filters import PrivacyConfig
    _ENHANCEMENTS_AVAILABLE = True
except ImportError as _e:
    print(f"[warn] Research enhancements not loaded: {_e}")
    _ENHANCEMENTS_AVAILABLE = False

try:
    from AutoWeb.src.policyHub.policy_hub import PolicyHub
    from shared.hitl.hitl_confidence import HITLConfidenceGate
    _HITL_AVAILABLE = True
except ImportError as _e:
    print(f"[warn] HITL modules not loaded: {_e}")
    _HITL_AVAILABLE = False

# -----------------------------
# Configuration (Notebook parity)
# -----------------------------
PLANNER_MODEL = os.getenv("PRUNE4WEB_PLANNER_MODEL", "gpt-4o")
FILTER_MODEL = os.getenv("PRUNE4WEB_FILTER_MODEL", "gpt-4o")
GROUNDER_MODEL = os.getenv("PRUNE4WEB_GROUNDER_MODEL", "gpt-4o")
MAX_STEPS = int(os.getenv("PRUNE4WEB_MAX_STEPS", "5"))
USE_VISION = os.getenv("PRUNE4WEB_USE_VISION", "1").strip().lower() in {"1", "true", "yes", "y"}
USE_HITL = os.getenv("PRUNE4WEB_USE_HITL", "1").strip().lower() in {"1", "true", "yes", "y"}

TOP_N_CANDIDATES = 20
FUZZY_THRESHOLD = 0.6

# --- Research enhancement flags ---
USE_DOM_DELTA   = os.getenv("PRUNE4WEB_DOM_DELTA", "1").strip().lower() in {"1", "true", "yes", "y"}
USE_PRIVACY     = os.getenv("PRUNE4WEB_PRIVACY", "1").strip().lower() in {"1", "true", "yes", "y"}
DP_EPSILON      = float(os.getenv("PRUNE4WEB_DP_EPSILON", "0"))  # 0 = DP off

HITL_THRESHOLD = int(os.getenv("PRUNE4WEB_HITL_THRESHOLD", os.getenv("HITL_THRESHOLD", "75")))
HITL_LOW_CONFIDENCE_FLOOR = int(
    os.getenv("PRUNE4WEB_HITL_LOW_CONFIDENCE_FLOOR", os.getenv("LOW_CONFIDENCE_FLOOR", "30"))
)
POLICY_HUB_PATH = os.getenv(
    "PRUNE4WEB_POLICY_HUB_PATH",
    str(Path(_PROJECT_ROOT) / "AutoWeb" / "src" / "policyHub" / "policies.json"),
)


ALPHA_EXACT = 4.0
ALPHA_PHRASE = 3.0
ALPHA_WORD = 2.0
ALPHA_FUZZY = 1.0

BETA_VISIBLE_TEXT = 3.0
BETA_ARIA_LABEL = 2.5
BETA_PLACEHOLDER = 2.0
BETA_ID_CLASS = 1.5
BETA_OTHER = 1.0

MODEL_PRICING_PER_1M = {
    # Override if needed for your account/region.
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
}

_CLIENT: Optional[OpenAI] = None


def get_client() -> OpenAI:
    if _CLIENT is None:
        raise RuntimeError("OpenAI client is not initialized.")
    return _CLIENT


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int) -> Optional[float]:
    pricing = MODEL_PRICING_PER_1M.get(model)
    if not pricing:
        return None
    non_cached_prompt = max(prompt_tokens - cached_tokens, 0)
    return (
        (non_cached_prompt / 1_000_000) * pricing["input"]
        + (cached_tokens / 1_000_000) * pricing["cached_input"]
        + (completion_tokens / 1_000_000) * pricing["output"]
    )


@dataclass
class LLMCallUsage:
    stage: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    total_tokens: int
    estimated_cost_usd: Optional[float]


class UsageTracker:
    def __init__(self) -> None:
        self.calls: List[LLMCallUsage] = []

    def add(self, stage: str, model: str, usage_obj) -> None:
        prompt = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage_obj, "completion_tokens", 0) or 0)
        total = int(getattr(usage_obj, "total_tokens", prompt + completion) or (prompt + completion))
        prompt_details = getattr(usage_obj, "prompt_tokens_details", None)
        cached = int(getattr(prompt_details, "cached_tokens", 0) or 0)
        cost = estimate_cost_usd(model, prompt, completion, cached)
        self.calls.append(
            LLMCallUsage(
                stage=stage,
                model=model,
                prompt_tokens=prompt,
                completion_tokens=completion,
                cached_tokens=cached,
                total_tokens=total,
                estimated_cost_usd=cost,
            )
        )
        print(
            f"  [usage:{stage}] model={model} prompt={prompt} completion={completion} "
            f"cached={cached} total={total} est_cost="
            f"{'$' + format(cost, '.6f') if cost is not None else 'n/a'}"
        )

    def print_summary(self) -> None:
        print("\n" + "=" * 72)
        print("LLM USAGE SUMMARY")
        print("=" * 72)
        if not self.calls:
            print("No model calls were recorded.")
            return
        total_prompt = sum(c.prompt_tokens for c in self.calls)
        total_completion = sum(c.completion_tokens for c in self.calls)
        total_cached = sum(c.cached_tokens for c in self.calls)
        total_tokens = sum(c.total_tokens for c in self.calls)
        total_cost = sum(c.estimated_cost_usd or 0.0 for c in self.calls)
        for i, c in enumerate(self.calls, start=1):
            cost_txt = f"${c.estimated_cost_usd:.6f}" if c.estimated_cost_usd is not None else "n/a"
            print(
                f"{i:02d}. stage={c.stage:<8} model={c.model:<10} "
                f"prompt={c.prompt_tokens:<6} completion={c.completion_tokens:<6} "
                f"cached={c.cached_tokens:<6} total={c.total_tokens:<6} cost={cost_txt}"
            )
        print("-" * 72)
        print(f"TOTAL prompt_tokens     : {total_prompt}")
        print(f"TOTAL completion_tokens : {total_completion}")
        print(f"TOTAL cached_tokens     : {total_cached}")
        print(f"TOTAL tokens            : {total_tokens}")
        print(f"TOTAL estimated cost    : ${total_cost:.6f}")
        print("=" * 72)


USAGE_TRACKER = UsageTracker()


def llm_call(messages, stage: str, model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    model = model or PLANNER_MODEL
    response = get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    USAGE_TRACKER.add(stage=stage, model=model, usage_obj=response.usage)
    return (response.choices[0].message.content or "").strip()


def encode_image(path_or_url: str) -> str:
    if path_or_url.startswith("http"):
        data = requests.get(path_or_url, timeout=15).content
    else:
        with open(path_or_url, "rb") as f:
            data = f.read()
    return base64.b64encode(data).decode("utf-8")


def parse_json_from_llm(text: str) -> Dict:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    raw = match.group(1).strip() if match else text.strip()
    raw = re.sub(r"^[^\[{]*", "", raw)
    raw = re.sub(r"[^\]\}]*$", "", raw)
    return json.loads(raw)


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(value, (int, float)):
        return value != 0
    return False


def _to_confidence_0_100(value, default: float = 50.0) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        conf = float(default)
    if conf <= 1.0:
        conf *= 100.0
    return max(0.0, min(100.0, conf))


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().strip()
    except Exception:
        return ""


INTERACTIVE_TAGS = {"a", "button", "input", "select", "textarea", "label", "option", "details", "summary"}
INTERACTIVE_ROLES = {
    "button",
    "link",
    "checkbox",
    "radio",
    "textbox",
    "combobox",
    "listbox",
    "menuitem",
    "tab",
    "switch",
    "searchbox",
    "spinbutton",
    "slider",
}


@dataclass
class ElementNode:
    uid: int
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

    def attribute_tuples(self) -> List[Tuple[str, float]]:
        pairs: List[Tuple[str, float]] = []
        if self.text:
            pairs.append((self.text.lower(), BETA_VISIBLE_TEXT))
        if self.aria_label:
            pairs.append((self.aria_label.lower(), BETA_ARIA_LABEL))
        if self.title:
            pairs.append((self.title.lower(), BETA_ARIA_LABEL))
        if self.placeholder:
            pairs.append((self.placeholder.lower(), BETA_PLACEHOLDER))
        if self.elem_id:
            pairs.append((self.elem_id.lower(), BETA_ID_CLASS))
        if self.name:
            pairs.append((self.name.lower(), BETA_ID_CLASS))
        for cls in self.elem_class.lower().split():
            pairs.append((cls, BETA_ID_CLASS))
        if self.href:
            pairs.append((self.href.lower(), BETA_OTHER))
        if self.value:
            pairs.append((self.value.lower(), BETA_OTHER))
        return pairs

    def to_summary(self) -> str:
        parts = [f"[{self.uid}] <{self.tag}"]
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


def parse_dom(html: str) -> List[ElementNode]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "meta", "noscript", "head"]):
        tag.decompose()

    nodes: List[ElementNode] = []
    uid = 0
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

        nodes.append(
            ElementNode(
                uid=uid,
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
            )
        )
        uid += 1
    return nodes


def _match_alpha(keyword: str, attr_text: str) -> float:
    k = keyword.strip().lower()
    a = attr_text.strip().lower()
    if not k or not a:
        return 0.0
    if k == a:
        return ALPHA_EXACT
    if " " in k and k in a:
        return ALPHA_PHRASE
    tokens = re.split(r"[\s\-_/]+", a)
    if k in tokens:
        return ALPHA_WORD
    ratio = fuzz.partial_ratio(k, a) / 100.0
    if ratio >= FUZZY_THRESHOLD:
        return ALPHA_FUZZY * ratio
    return 0.0


def score_elements(elements: List[ElementNode], keyword_weights: Dict[str, float], top_n: int = TOP_N_CANDIDATES) -> List[ElementNode]:
    scores: Dict[int, float] = {}
    for element in elements:
        score = 0.0
        for attr_text, beta in element.attribute_tuples():
            for keyword, w_base in keyword_weights.items():
                alpha = _match_alpha(keyword, attr_text)
                if alpha > 0.0:
                    score += w_base * alpha * beta
        scores[element.uid] = score

    sorted_uids = sorted(scores, key=lambda uid: -scores[uid])
    uid_map = {element.uid: element for element in elements}
    return [uid_map[uid] for uid in sorted_uids[:top_n]]


PLANNER_SYSTEM = """\
You are the Planner of Prune4Web, a web automation agent.

Given a high-level task and current page state, output ONE low-level sub-task
for this step as a JSON object with exactly these fields:
{
  "sub_task": "<short imperative sentence identifying the target element semantically>",
  "action_type": "<click|type|select|scroll|hover|check>",
  "value": "<text to type or option; empty string if not applicable>",
  "reasoning": "<=2 sentence chain-of-thought>",
  "confidence": <integer 0-100>,
  "policy_risk": <true|false>,
  "hitl_reason": "<short reason, empty string if no risk>"
}
Return ONLY the JSON. No markdown fences. No preamble.
"""

FILTER_SYSTEM = """\
You are the Programmatic Element Filter of Prune4Web.

Given a low-level web sub-task, output a JSON mapping of semantic keywords
to base weights so a Python scoring script can locate the correct DOM element.

Rules:
  - 3-10 discriminative keywords
  - Weights: positive floats 0.1 to 3.0 (higher = more important)
  - Include synonyms at lower weights
  - IMPORTANT synonym expansions you must always apply:
      "add to cart" → also include: "bag", "basket", "add" (Apple uses "Add to Bag")
      "buy" / "purchase" → also include: "shop", "order"
      "search" → also include: "find", "query", "lookup"

Example: {"destination": 2.0, "city": 1.0, "flight": 1.5, "to": 0.5}

Return ONLY the JSON object. No markdown. No preamble.
"""

GROUNDER_SYSTEM = """\
You are the Action Grounder of Prune4Web.

Given a low-level sub-task and a small ranked list of candidate DOM elements,
identify exactly which element to interact with.

IMPORTANT label equivalences — treat these as identical:
  "Add to Cart" = "Add to Bag" = "Add to Basket" = "Buy" (Apple uses "Add to Bag")
  "Cart" = "Bag" = "Basket"

Return ONLY a JSON object:
{
  "element_uid": <integer UID>,
  "action": "<click|type|select|scroll|hover|check>",
  "value": "<text to input; empty string if not applicable>",
  "confidence": <float 0-1>,
  "reasoning": "<=2 sentences"
}
If no candidate matches, use element_uid: -1.
No markdown fences. No preamble. Pure JSON only.
"""


def planner(task: str, screenshot_path: Optional[str], history: List[str], page_title: str) -> Dict:
    hist_str = "\n".join(f"  Step {i + 1}: {value}" for i, value in enumerate(history)) or "  (none)"
    user_text = f"High-level task: {task}\nSteps completed:\n{hist_str}\nPage title: {page_title}"
    messages = [{"role": "system", "content": PLANNER_SYSTEM}]

    if screenshot_path and USE_VISION:
        b64 = encode_image(screenshot_path)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        )
        try:
            raw = llm_call(messages=messages, stage="planner", model=PLANNER_MODEL, max_tokens=512)
        except BadRequestError as exc:
            print(f"  Planner vision request failed, retrying text-only: {exc}")
            messages = [{"role": "system", "content": PLANNER_SYSTEM}, {"role": "user", "content": user_text}]
            raw = llm_call(messages=messages, stage="planner", model=PLANNER_MODEL, max_tokens=512)
    else:
        messages.append({"role": "user", "content": user_text})
        raw = llm_call(messages=messages, stage="planner", model=PLANNER_MODEL, max_tokens=512)
    try:
        return json.loads(raw)
    except Exception:
        return parse_json_from_llm(raw)


def generate_keyword_weights(sub_task: str) -> Dict[str, float]:
    messages = [
        {"role": "system", "content": FILTER_SYSTEM},
        {"role": "user", "content": f"Sub-task: {sub_task}"},
    ]
    raw = llm_call(messages=messages, stage="filter", model=FILTER_MODEL, max_tokens=256)
    try:
        keywords = json.loads(raw)
    except Exception:
        keywords = parse_json_from_llm(raw)
    return {str(k): float(v) for k, v in keywords.items()}


def programmatic_element_filter(sub_task: str, elements: List[ElementNode], top_n: int = TOP_N_CANDIDATES) -> Tuple[List[ElementNode], Dict[str, float]]:
    keywords = generate_keyword_weights(sub_task)
    candidates = score_elements(elements, keywords, top_n=top_n)
    return candidates, keywords


def action_grounder(sub_task: str, candidates: List[ElementNode], action_value: str = "") -> Dict:
    candidate_lines = [f"  {i + 1}. {el.to_summary()}" for i, el in enumerate(candidates)]
    user_msg = (
        f"Sub-task: {sub_task}\nValue hint: {action_value}\n\n"
        f"Candidate elements (ranked by relevance):\n" + "\n".join(candidate_lines)
    )
    messages = [
        {"role": "system", "content": GROUNDER_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    raw = llm_call(messages=messages, stage="grounder", model=GROUNDER_MODEL, max_tokens=512)
    try:
        return json.loads(raw)
    except Exception:
        return parse_json_from_llm(raw)


# -----------------------------
# Optional: local fine-tuned grounder
# -----------------------------
# Set GROUNDER_USE_LOCAL=1 to swap the OpenAI grounder for the locally
# fine-tuned Qwen2.5-0.5B LoRA adapter produced by
# Prune4Web/grounder_finetune/scripts/02_finetune.py.
if os.getenv("GROUNDER_USE_LOCAL", "").strip() in {"1", "true", "yes"}:
    try:
        from Prune4Web.grounder_finetune.local_grounder import ground as _local_ground
        print("[grounder] Using LOCAL fine-tuned Qwen2.5-0.5B (GROUNDER_USE_LOCAL=1)")
        action_grounder = _local_ground  # type: ignore[assignment]
    except Exception as _lg_exc:
        print(f"[grounder] Failed to load local grounder ({_lg_exc}); "
              f"falling back to OpenAI.")


@dataclass
class StepResult:
    step_idx: int
    sub_task: str
    action_type: str
    keyword_weights: Dict[str, float]
    num_dom_nodes: int
    num_candidates: int
    grounded_uid: int
    grounded_element_summary: str
    action_value: str
    llm_confidence: float
    composite_confidence: float
    policy_risk_flag: bool
    hitl_triggered: bool
    hitl_reason: str
    hitl_details: Dict
    confidence: float
    reasoning: str
    executed: bool
    executed_selector: str
    grounding_error: str


def _clean_action(action: str, fallback: str) -> str:
    allowed = {"click", "type", "select", "scroll", "hover", "check"}
    candidate = (action or "").strip().lower()
    if candidate in allowed:
        return candidate
    fallback = (fallback or "").strip().lower()
    return fallback if fallback in allowed else "click"


async def _highlight_and_scroll(page: Page, js_id: str) -> None:
    """Scroll element into view and flash a red border. Uses JS getElementById — works with any ID."""
    try:
        await page.evaluate(
            """(id) => {
                const el = document.getElementById(id);
                if (!el) return;
                el.scrollIntoView({behavior: 'smooth', block: 'center'});
                const prev = el.style.outline;
                el.style.outline = '3px solid #FF3B30';
                el.style.outlineOffset = '2px';
                setTimeout(() => { el.style.outline = prev; }, 1200);
            }""",
            js_id,
        )
    except Exception:
        pass


async def _do_action_on_loc(loc, action: str, value: str) -> None:
    """Execute a single action on a Playwright locator."""
    if action == "click":
        await loc.click(timeout=5000)
    elif action == "type":
        await loc.fill(value, timeout=5000)
    elif action == "select":
        try:
            await loc.select_option(value=value, timeout=5000)
        except Exception:
            await loc.select_option(label=value, timeout=5000)
    elif action == "hover":
        await loc.hover(timeout=5000)
    elif action == "check":
        await loc.check(timeout=5000)
    elif action == "scroll":
        await loc.scroll_into_view_if_needed(timeout=5000)
        await loc.page.evaluate("window.scrollBy({top:200, behavior:'smooth'})")
    else:
        await loc.click(timeout=5000)


async def _execute_action(page: Page, element: Optional[ElementNode], action: str, value: str) -> Tuple[bool, str]:
    """
    Multi-strategy execution engine.

    Tries 8 strategies in priority order, returns (True, strategy_label) on
    first success. Handles React/Next.js dynamic IDs (e.g. ':r8:') that break
    standard CSS selectors by using JavaScript getElementById as Strategy 1.
    """
    # --- Scroll with no target element: page-level scroll -------------------
    if action == "scroll" and element is None:
        await page.evaluate("window.scrollBy({top: 500, behavior: 'smooth'})")
        print("  Executed scroll (page-level)")
        return True, "page-scroll"

    if element is None:
        print("  Execution skipped: no grounded element.")
        return False, ""

    # Helper: try a locator, highlight it, then act
    async def _try(loc, label: str) -> bool:
        try:
            await loc.wait_for(state="visible", timeout=2500)
            # Highlight via JS so special-char IDs don't matter
            try:
                bbox = await loc.bounding_box()
                if bbox:
                    await page.evaluate(
                        """([x,y,w,h]) => {
                            const el = document.elementFromPoint(x+w/2, y+h/2);
                            if(!el) return;
                            const p=el.style.outline; el.style.outline='3px solid #FF3B30';
                            setTimeout(()=>{el.style.outline=p;},1200);
                        }""",
                        [bbox["x"], bbox["y"], bbox["width"], bbox["height"]],
                    )
            except Exception:
                pass
            await page.wait_for_timeout(350)
            await _do_action_on_loc(loc, action, value)
            print(f"  Executed {action}('{value}') via [{label}]")
            return True
        except Exception:
            return False

    elem_id    = element.elem_id or ""
    elem_text  = (element.text or "").strip()
    aria       = element.aria_label or ""
    placeholder = element.placeholder or ""
    name       = element.name or ""
    tag        = element.tag or ""
    is_radio   = element.input_type == "radio"

    # ------------------------------------------------------------------
    # Strategy 1 – JavaScript getElementById (bypasses ALL CSS escaping)
    #   For radio buttons: prefer clicking the associated <label> via JS
    # ------------------------------------------------------------------
    if elem_id:
        try:
            clicked = await page.evaluate(
                """(id) => {
                    const el = document.getElementById(id);
                    if (!el) return false;
                    // For radio inputs, try clicking the paired label first
                    if (el.type === 'radio') {
                        const lbl = document.querySelector('label[for="' + id + '"]')
                               || el.closest('label');
                        if (lbl) { lbl.scrollIntoView({block:'center'}); lbl.click(); return true; }
                    }
                    el.scrollIntoView({behavior:'smooth', block:'center'});
                    const prev = el.style.outline;
                    el.style.outline = '3px solid #FF3B30';
                    el.style.outlineOffset = '2px';
                    setTimeout(() => { el.style.outline = prev; }, 1200);
                    el.click();
                    return true;
                }""",
                elem_id,
            )
            if clicked:
                print(f"  Executed {action}('{value}') via [js-id:{elem_id}]")
                await page.wait_for_timeout(350)
                # For type/select we still need Playwright after JS scroll+highlight
                if action in ("type", "select"):
                    loc = page.locator(f"[id]").filter(has_text="").nth(0)  # will fail → fallthrough
                    # Use attribute selector with double-quotes (safe for colons)
                    attr_loc = page.locator(f'[id="{elem_id}"]').first
                    if action == "type":
                        await attr_loc.fill(value, timeout=4000)
                    elif action == "select":
                        try:
                            await attr_loc.select_option(value=value, timeout=4000)
                        except Exception:
                            await attr_loc.select_option(label=value, timeout=4000)
                return True, f"js-id:{elem_id}"
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Strategy 2 – Playwright get_by_text (exact visible text)
    # ------------------------------------------------------------------
    short_text = elem_text[:80] if elem_text else ""
    if short_text and len(short_text) <= 80 and not is_radio:
        loc = page.get_by_text(short_text, exact=True).first
        if await _try(loc, f"text:{short_text[:30]}"):
            return True, f"text:{short_text[:30]}"

    # ------------------------------------------------------------------
    # Strategy 3 – get_by_label (pairs inputs with their label text)
    # ------------------------------------------------------------------
    label_text = aria or short_text or placeholder
    if label_text and tag in ("input", "select", "textarea"):
        loc = page.get_by_label(label_text, exact=True).first
        if await _try(loc, f"label:{label_text[:30]}"):
            return True, f"label:{label_text[:30]}"

    # ------------------------------------------------------------------
    # Strategy 4 – get_by_role + accessible name
    # ------------------------------------------------------------------
    role_map = {
        "button": "button", "a": "link", "input": None,
        "select": "combobox", "textarea": "textbox",
        "label": None, "option": "option",
    }
    aria_role = element.role or role_map.get(tag)
    if aria_role and (aria or short_text):
        name_hint = aria or short_text
        try:
            loc = page.get_by_role(aria_role, name=name_hint).first
            if await _try(loc, f"role:{aria_role}+name"):
                return True, f"role:{aria_role}"
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Strategy 5 – [aria-label="..."] attribute selector
    # ------------------------------------------------------------------
    if aria:
        safe = aria.replace('"', '\\"')
        loc = page.locator(f'[aria-label="{safe}"]').first
        if await _try(loc, f"aria-label:{aria[:30]}"):
            return True, f"aria-label:{aria[:30]}"

    # ------------------------------------------------------------------
    # Strategy 6 – [placeholder="..."] attribute selector
    # ------------------------------------------------------------------
    if placeholder:
        safe = placeholder.replace('"', '\\"')
        loc = page.locator(f'[placeholder="{safe}"]').first
        if await _try(loc, f"placeholder:{placeholder[:30]}"):
            return True, f"placeholder:{placeholder[:30]}"

    # ------------------------------------------------------------------
    # Strategy 7 – [name="..."] attribute selector
    # ------------------------------------------------------------------
    if name:
        safe = name.replace('"', '\\"')
        loc = page.locator(f'[name="{safe}"]').first
        if await _try(loc, f"name:{name}"):
            return True, f"name:{name}"

    # ------------------------------------------------------------------
    # Strategy 8 – JavaScript querySelectorAll by text content (last resort)
    # ------------------------------------------------------------------
    if short_text:
        try:
            clicked = await page.evaluate(
                """(txt) => {
                    const all = document.querySelectorAll('button,a,label,input,select');
                    for (const el of all) {
                        if ((el.textContent||'').trim() === txt ||
                            (el.getAttribute('aria-label')||'') === txt) {
                            el.scrollIntoView({block:'center'});
                            el.click();
                            return true;
                        }
                    }
                    return false;
                }""",
                short_text,
            )
            if clicked:
                print(f"  Executed {action}('{value}') via [js-text:{short_text[:30]}]")
                return True, f"js-text:{short_text[:30]}"
        except Exception:
            pass

    print("  Could not execute action — all 8 strategies failed.")
    return False, ""


def print_step_results(results: List[StepResult]) -> None:
    print("\n" + "=" * 72)
    print("PRUNE4WEB STEP RESULTS")
    print("=" * 72)
    for result in results:
        reduction = result.num_dom_nodes / max(result.num_candidates, 1)
        print(f"Step {result.step_idx}")
        print(f"  sub_task      : {result.sub_task}")
        print(f"  action        : {result.action_type}('{result.action_value}')")
        print(f"  keywords      : {result.keyword_weights}")
        print(f"  dom_filter    : {result.num_dom_nodes} -> {result.num_candidates} ({reduction:.1f}x)")
        print(f"  llm_conf      : {result.llm_confidence:.1f}")
        print(f"  composite_conf: {result.composite_confidence:.1f}")
        print(f"  policy_risk   : {result.policy_risk_flag}")
        print(f"  hitl_triggered: {result.hitl_triggered}")
        if result.hitl_reason:
            print(f"  hitl_reason   : {result.hitl_reason}")
        print(f"  grounded_uid  : {result.grounded_uid}")
        print(f"  grounded_elem : {result.grounded_element_summary}")
        print(f"  confidence    : {result.confidence:.3f}")
        print(f"  reasoning     : {result.reasoning}")
        print(f"  executed      : {result.executed} ({result.executed_selector or 'none'})")
        if result.grounding_error:
            print(f"  grounding_err : {result.grounding_error}")
        print("-" * 72)
    if results:
        avg_reduction = sum(r.num_dom_nodes / max(r.num_candidates, 1) for r in results) / len(results)
        print(f"Average DOM reduction: {avg_reduction:.1f}x")
        print(f"Total steps         : {len(results)}")
    print("=" * 72)


async def run_prune4web_live(
    url: str,
    task: str,
    max_steps: int = MAX_STEPS,
    headless: bool = False,
    website_domain: str = "",
) -> List[StepResult]:
    results: List[StepResult] = []
    history: List[str] = []
    screenshot_dir = Path(tempfile.gettempdir())
    domain = website_domain or _extract_domain(url)

    policy_hub = None
    hitl_gate = None
    policy_risk_keywords: List[str] = []
    if USE_HITL and _HITL_AVAILABLE:
        policy_hub = PolicyHub(json_path=POLICY_HUB_PATH)
        hitl_gate = HITLConfidenceGate(
            low_confidence_floor=HITL_LOW_CONFIDENCE_FLOOR,
            threshold=HITL_THRESHOLD,
        )
        policy_risk_keywords = policy_hub.get_policy_risk_keywords()
        print(
            "  [hitl] ENABLED "
            f"(domain={domain or 'n/a'}, low_conf_floor={HITL_LOW_CONFIDENCE_FLOOR}, "
            f"keywords={len(policy_risk_keywords)})"
        )
    elif USE_HITL:
        print("  [hitl] requested but unavailable; continuing without HITL")

    # --- Research enhancement setup ----------------------------------------
    delta_processor = None
    privacy_llm: Optional[PrivacyAwareLLM] = None  # type: ignore[name-defined]

    if _ENHANCEMENTS_AVAILABLE:
        if USE_DOM_DELTA:
            delta_processor = DOMDeltaProcessor()
            print("  [enhancement] DOM Delta Processing: ENABLED")
        if USE_PRIVACY:
            privacy_llm = build_privacy_aware_llm(
                raw_llm_call=llm_call,
                enable_dp=DP_EPSILON > 0,
                dp_epsilon=DP_EPSILON if DP_EPSILON > 0 else 1.0,
            )
            print("  [enhancement] Privacy-Aware LLM: ENABLED")
    # -----------------------------------------------------------------------

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=headless)
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()

        print(f"\nNavigating to: {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_timeout(1500)

        _stall_key = ""     # tracks "(sub_task|uid=N)" — both must match to count as stall
        _stall_count = 0
        _STALL_LIMIT = 3    # stop after this many consecutive identical failures

        for step_idx in range(1, max_steps + 1):
            screenshot_path = screenshot_dir / f"prune4web_step_{step_idx}.png"
            await page.screenshot(path=str(screenshot_path), full_page=False)
            html = await page.content()
            title = await page.title()
            url_now = page.url
            elements = parse_dom(html)

            print("\n" + "-" * 72)
            print(f"STEP {step_idx}")
            print(f"  page_title      : {title}")
            print(f"  screenshot_path : {screenshot_path}")
            print(f"  html_chars      : {len(html)}")
            print(f"  dom_nodes       : {len(elements)}")

            # --- DOM Delta: update snapshot and compute delta ---------------
            delta_elements = None
            delta_info = ""
            if delta_processor is not None:
                dom_state, delta = delta_processor.update_and_diff(
                    html, url=url_now, title=title, timestamp=_time.time()
                )
                if delta is not None:
                    delta.print_summary()
                    # Convert SnapshotElements to ElementNodes for compatibility
                    # (use full elements on first step or if delta is tiny)
                    snap_relevant = delta_processor.get_relevant_elements(
                        delta, task_context=task
                    )
                    # Map snapshot uid → ElementNode by text/id similarity
                    snap_ids = {s.elem_id for s in snap_relevant if s.elem_id}
                    snap_texts = {s.text[:60] for s in snap_relevant if s.text}
                    delta_elements = [
                        el for el in elements
                        if el.elem_id in snap_ids
                        or el.text[:60] in snap_texts
                    ] or elements  # fallback to full set if no matches
                    delta_info = (
                        f"delta_relevant={len(delta_elements)}/{len(elements)} "
                        f"(reduction={len(elements)/max(len(delta_elements),1):.1f}x)"
                    )
                    print(f"  dom_delta       : {delta_info}")
            # Use delta-filtered elements if available, else full set
            candidate_source = delta_elements if delta_elements is not None else elements
            # ---------------------------------------------------------------

            plan = planner(task=task, screenshot_path=str(screenshot_path), history=history, page_title=title)
            sub_task = str(plan.get("sub_task", "")).strip()
            planned_action = str(plan.get("action_type", "click")).strip()
            value_hint = str(plan.get("value", "")).strip()
            planner_reason = str(plan.get("reasoning", "")).strip()
            llm_confidence = _to_confidence_0_100(plan.get("confidence", 50.0))
            policy_risk_flag = _to_bool(plan.get("policy_risk", False))
            hitl_reason_from_llm = str(plan.get("hitl_reason", "")).strip()

            print(f"  planner_sub_task: {sub_task}")
            print(f"  planner_action  : {planned_action}")
            print(f"  planner_value   : {value_hint}")
            print(f"  planner_reason  : {planner_reason}")
            print(f"  planner_conf    : {llm_confidence:.1f}")
            print(f"  planner_policy  : {policy_risk_flag}")
            if hitl_reason_from_llm:
                print(f"  planner_hitl    : {hitl_reason_from_llm}")

            if not sub_task:
                print("  Planner did not return a sub-task. Stopping.")
                break
            if any(marker in sub_task.lower() for marker in ["task complete", "already done", "finished", "done"]):
                print("  Planner signaled task completion.")
                break

            hitl_details: Dict = {}
            composite_confidence = llm_confidence
            hitl_triggered = False
            hitl_reason = hitl_reason_from_llm
            if hitl_gate is not None:
                action_text = f"{sub_task} -> {planned_action} {value_hint}".strip()
                hitl_details = hitl_gate.compute_composite_confidence(
                    llm_confidence=llm_confidence,
                    action_text=action_text,
                    policy_risk_keywords=policy_risk_keywords,
                    policy_risk_flag=policy_risk_flag,
                    hitl_reason=hitl_reason_from_llm,
                )
                composite_confidence = float(hitl_details.get("final_confidence", llm_confidence))
                hitl_triggered = bool(hitl_details.get("hitl_triggered", False))
                if not hitl_reason:
                    hitl_reason = "; ".join(hitl_details.get("trigger_reasons", []))
                print(
                    f"  hitl_gate       : triggered={hitl_triggered} "
                    f"reasons={hitl_details.get('trigger_reasons', [])}"
                )

            if hitl_triggered:
                action = _clean_action(planned_action, "click")
                step_result = StepResult(
                    step_idx=step_idx,
                    sub_task=sub_task,
                    action_type=action,
                    keyword_weights={},
                    num_dom_nodes=len(elements),
                    num_candidates=0,
                    grounded_uid=-1,
                    grounded_element_summary="(skipped: HITL triggered)",
                    action_value=value_hint,
                    llm_confidence=llm_confidence,
                    composite_confidence=composite_confidence,
                    policy_risk_flag=policy_risk_flag,
                    hitl_triggered=True,
                    hitl_reason=hitl_reason,
                    hitl_details=hitl_details,
                    confidence=0.0,
                    reasoning="Grounding skipped due to HITL trigger.",
                    executed=False,
                    executed_selector="",
                    grounding_error="hitl_triggered",
                )
                results.append(step_result)
                history.append(f"{sub_task} -> HITL(triggered) action={action}")
                print("  HITL triggered: skipping filter/grounding/execution and stopping for human review.")
                break

            candidates, keywords = programmatic_element_filter(
                sub_task=sub_task, elements=candidate_source, top_n=TOP_N_CANDIDATES
            )
            print(f"  keyword_weights : {keywords}")
            print(f"  candidates_topN : {len(candidates)} / {TOP_N_CANDIDATES}")

            grounded = action_grounder(sub_task=sub_task, candidates=candidates, action_value=value_hint)
            grounded_uid = int(grounded.get("element_uid", -1))
            action = _clean_action(str(grounded.get("action", planned_action)), planned_action)
            action_value = str(grounded.get("value", value_hint))
            confidence = float(grounded.get("confidence", 0.0) or 0.0)
            reasoning = str(grounded.get("reasoning", ""))
            matched = next((element for element in candidates if element.uid == grounded_uid), None)

            print(f"  grounded_uid    : {grounded_uid}")
            print(f"  grounded_action : {action}")
            print(f"  grounded_value  : {action_value}")
            print(f"  grounded_conf   : {confidence}")
            print(f"  grounded_reason : {reasoning}")

            executed, selector = await _execute_action(page, matched, action, action_value)

            step_result = StepResult(
                step_idx=step_idx,
                sub_task=sub_task,
                action_type=action,
                keyword_weights=keywords,
                num_dom_nodes=len(elements),
                num_candidates=len(candidates),
                grounded_uid=grounded_uid,
                grounded_element_summary=matched.to_summary() if matched else "(not found in top candidates)",
                action_value=action_value,
                llm_confidence=llm_confidence,
                composite_confidence=composite_confidence,
                policy_risk_flag=policy_risk_flag,
                hitl_triggered=False,
                hitl_reason=hitl_reason,
                hitl_details=hitl_details,
                confidence=confidence,
                reasoning=reasoning,
                executed=executed,
                executed_selector=selector,
                grounding_error="",
            )
            results.append(step_result)
            history.append(f"{sub_task} -> {action}('{action_value}') uid={grounded_uid} executed={executed}")

            # --- Stall detection -------------------------------------------
            current_key = f"{sub_task}|uid={grounded_uid}"
            if not executed and current_key == _stall_key:
                _stall_count += 1
                if _stall_count >= _STALL_LIMIT:
                    print(f"\n  [stall] '{sub_task}' (uid={grounded_uid}) failed "
                          f"{_STALL_LIMIT}x in a row. Stopping early.")
                    break
            else:
                _stall_count = 0 if executed else 1
            _stall_key = current_key
            # ---------------------------------------------------------------

            await page.wait_for_timeout(1000)

        await context.close()
        await browser.close()

    # --- Print enhancement summaries ---------------------------------------
    if privacy_llm:
        privacy_llm.print_privacy_summary()

    return results


def _ask_non_empty(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Input cannot be empty.")


def print_runtime_parameters(url: str, task: str, max_steps: int) -> None:
    print("=" * 72)
    print("RUNTIME PARAMETERS")
    print("=" * 72)
    print(f"url                 : {url}")
    print(f"task                : {task}")
    print(f"planner_model       : {PLANNER_MODEL}")
    print(f"filter_model        : {FILTER_MODEL}")
    print(f"grounder_model      : {GROUNDER_MODEL}")
    print(f"max_steps           : {max_steps}")
    print(f"headless            : False (live browser preview enabled)")
    print(f"top_n_candidates    : {TOP_N_CANDIDATES}")
    print(f"fuzzy_threshold     : {FUZZY_THRESHOLD}")
    print(f"alpha_exact         : {ALPHA_EXACT}")
    print(f"alpha_phrase        : {ALPHA_PHRASE}")
    print(f"alpha_word          : {ALPHA_WORD}")
    print(f"alpha_fuzzy         : {ALPHA_FUZZY}")
    print(f"beta_visible_text   : {BETA_VISIBLE_TEXT}")
    print(f"beta_aria_label     : {BETA_ARIA_LABEL}")
    print(f"beta_placeholder    : {BETA_PLACEHOLDER}")
    print(f"beta_id_class       : {BETA_ID_CLASS}")
    print(f"beta_other          : {BETA_OTHER}")
    print(f"dom_delta_enabled   : {USE_DOM_DELTA and _ENHANCEMENTS_AVAILABLE}")
    print(f"privacy_enabled     : {USE_PRIVACY and _ENHANCEMENTS_AVAILABLE}")
    print(f"dp_epsilon          : {DP_EPSILON if DP_EPSILON > 0 else 'disabled'}")
    print(f"hitl_enabled        : {USE_HITL and _HITL_AVAILABLE}")
    print(f"hitl_low_conf_floor : {HITL_LOW_CONFIDENCE_FLOOR}")
    print(f"hitl_threshold      : {HITL_THRESHOLD}")
    print(f"policy_hub_path     : {POLICY_HUB_PATH}")
    print("=" * 72)


def _save_run_log(url: str, task: str, max_steps: int, results: List[StepResult]) -> None:
    """Save JSON + plain-text .log artifacts for each live run."""
    import datetime
    run_dir = Path(__file__).resolve().parent / "results" / "live_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize URL for filename
    domain = url.split("//")[-1].split("/")[0].replace(".", "_").replace(":", "")
    base_name = f"{ts}_{domain}"
    filename = f"{base_name}.json"

    use_local = os.getenv("GROUNDER_USE_LOCAL", "").strip() in {"1", "true", "yes"}
    steps_data = []
    for r in results:
        steps_data.append({
            "step": r.step_idx,
            "sub_task": r.sub_task,
            "action": r.action_type,
            "action_value": r.action_value,
            "keyword_weights": r.keyword_weights,
            "dom_nodes": r.num_dom_nodes,
            "candidates": r.num_candidates,
            "grounded_uid": r.grounded_uid,
            "grounded_element": r.grounded_element_summary,
            "llm_confidence": r.llm_confidence,
            "composite_confidence": r.composite_confidence,
            "policy_risk_flag": r.policy_risk_flag,
            "hitl_triggered": r.hitl_triggered,
            "hitl_reason": r.hitl_reason,
            "hitl_details": r.hitl_details,
            "confidence": r.confidence,
            "reasoning": r.reasoning,
            "executed": r.executed,
            "executed_selector": r.executed_selector,
            "grounding_error": r.grounding_error,
        })

    usage_data = []
    for c in USAGE_TRACKER.calls:
        usage_data.append({
            "stage": c.stage, "model": c.model,
            "prompt_tokens": c.prompt_tokens,
            "completion_tokens": c.completion_tokens,
            "cached_tokens": c.cached_tokens,
            "cost_usd": c.estimated_cost_usd,
        })

    total_cost = sum(c.estimated_cost_usd or 0.0 for c in USAGE_TRACKER.calls)
    executed_count = sum(1 for r in results if r.executed)
    hitl_count = sum(1 for r in results if r.hitl_triggered)
    domain_only = _extract_domain(url)

    run_log = {
        "timestamp": ts,
        "input": {
            "url": url,
            "domain": domain_only,
            "task": task,
            "max_steps": max_steps,
        },
        "config": {
            "planner_model": PLANNER_MODEL,
            "filter_model": FILTER_MODEL,
            "grounder_model": "local:Qwen2.5-0.5B-Prune4Web" if use_local else GROUNDER_MODEL,
            "grounder_local": use_local,
            "top_n_candidates": TOP_N_CANDIDATES,
            "dom_delta_enabled": USE_DOM_DELTA and _ENHANCEMENTS_AVAILABLE,
            "privacy_enabled": USE_PRIVACY and _ENHANCEMENTS_AVAILABLE,
            "hitl_enabled": USE_HITL and _HITL_AVAILABLE,
            "hitl_low_confidence_floor": HITL_LOW_CONFIDENCE_FLOOR,
            "hitl_threshold": HITL_THRESHOLD,
            "policy_hub_path": POLICY_HUB_PATH,
        },
        "output": {
            "total_steps": len(results),
            "steps_executed": executed_count,
            "steps_failed": len(results) - executed_count,
            "hitl_triggers": hitl_count,
            "hitl_trigger_rate": round(hitl_count / len(results), 4) if results else 0,
            "success_rate": f"{executed_count / len(results) * 100:.1f}%" if results else "0%",
            "total_api_cost_usd": round(total_cost, 6),
        },
        "steps": steps_data,
        "llm_usage": usage_data,
    }

    out_path = run_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2, ensure_ascii=False)

    txt_lines = [
        f"timestamp={ts}",
        f"url={url}",
        f"domain={domain_only}",
        f"task={task}",
        f"max_steps={max_steps}",
        f"total_steps={len(results)}",
        f"steps_executed={executed_count}",
        f"hitl_triggers={hitl_count}",
        f"api_cost_usd={round(total_cost, 6)}",
        "",
        "steps:",
    ]
    for r in results:
        txt_lines.append(
            f"step={r.step_idx} action={r.action_type} executed={r.executed} "
            f"hitl={r.hitl_triggered} llm_conf={r.llm_confidence:.1f} "
            f"comp_conf={r.composite_confidence:.1f} sub_task={r.sub_task}"
        )
        if r.hitl_reason:
            txt_lines.append(f"  hitl_reason={r.hitl_reason}")
        if r.grounding_error:
            txt_lines.append(f"  grounding_error={r.grounding_error}")

    txt_path = run_dir / f"{base_name}.log"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + "\n")

    print(f"\nRun log saved -> {out_path}")
    print(f"Text log saved -> {txt_path}")


def main() -> None:
    global _CLIENT
    # Ensure Proactor loop on Windows for Playwright subprocess support.
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Support CLI args for non-interactive use (e.g. from scripts/automation)
    import argparse as _ap
    _p = _ap.ArgumentParser(add_help=False)
    _p.add_argument("--url", default="")
    _p.add_argument("--task", default="")
    _p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    _args, _ = _p.parse_known_args()

    url = _args.url or _ask_non_empty("Enter URL for Prune4Web: ")
    task = _args.task or _ask_non_empty("Enter task for Prune4Web: ")
    max_steps = _args.max_steps
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Add it to your .env file:\n"
            "  OPENAI_API_KEY=sk-proj-..."
        )
    _CLIENT = OpenAI(api_key=api_key)

    print_runtime_parameters(url=url, task=task, max_steps=max_steps)
    results = asyncio.run(run_prune4web_live(url=url, task=task, max_steps=max_steps, headless=False))
    print_step_results(results)
    USAGE_TRACKER.print_summary()

    # ---- Save structured run log to results/live_runs/ ----
    _save_run_log(url=url, task=task, max_steps=max_steps, results=results)

    print("\nTask finished. Script is stopping.")


if __name__ == "__main__":
    main()
