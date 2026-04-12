# Input_Prepration.py

from typing import Dict, List, Tuple, Optional

class SeeActInputPreparator:    
    def __init__(self):
        pass

    def prepare_input(self,
                      action: Dict,
                      history: List[Dict],
                      policy_text: Optional[str] = None) -> Dict:
        """
        Prepare input for a single action generation step.
        
        Args:
            action: Dict containing details of the current action (including instruction, website screenshot, cleaned HTML, etc.)
            history: List of previous Action Generation planes/steps in the task (if needed for input preparation)
            policy_text: Optional policy constraints text from PolicyHub. When provided,
                         injected into the prompt and the output format switches to
                         structured JSON with confidence scoring.
        Returns:
            Dict containing prepared inputs for action generation, including:
            - "screenshot": website screenshot to be given in Action Generation
            - "prompt": the prompt text to be given in Action Generation (including instruction, history, etc.)
        """
        
        instruction = action.get("instruction", "")

        website_screenshot = action.get("screenshot", None)

        # Format history into a numbered list for clearer context (handles list or string inputs)
        history_text = ""
        try:
            if isinstance(history, list):
                lines = []
                for i, h in enumerate(history):
                    # extract readable text from dict-like entries
                    if isinstance(h, dict):
                        text = h.get("action_plan") or h.get("output_text") or h.get("raw_text") or str(h)
                    else:
                        text = str(h)

                    text = "\n".join(text.strip().splitlines())
                    lines.append(f"{i}. {text}")
                history_text = "\n".join(lines) if lines else "(none)"
            elif isinstance(history, str):
                parts = [p.strip() for p in history.splitlines() if p.strip()]
                if parts:
                    history_text = "\n".join(f"{i}. {p}" for i, p in enumerate(parts))
                else:
                    history_text = history
            else:
                history_text = str(history)
        except Exception:
            history_text = str(history)

        # Build optional policy constraints block
        policy_block = ""
        if policy_text and policy_text.strip():
            policy_block = (
                "\n"
                + self._separator() + "\n"
                + "POLICY CONSTRAINTS (you MUST follow these strictly):\n"
                + policy_text + "\n\n"
                + "RISK ASSESSMENT RULES:\n"
                + "- Set \"policy_risk\" to true ONLY when the action involves: "
                + "destructive operations (delete, remove, cancel), financial transactions "
                + "(purchase, payment, checkout, transfer), credentials (password, login), "
                + "CAPTCHA / security mechanisms, or account changes (unsubscribe, terminate, approve).\n"
                + "- Set \"policy_risk\" to false for normal navigation, reading, searching, "
                + "clicking links, typing search queries, or any non-destructive browsing.\n"
                + "- When \"policy_risk\" is true, explain why in \"hitl_reason\".\n"
                + "- Report REALISTIC confidence based on how certain you are about element selection. "
                + "Do NOT artificially reduce confidence for safe actions just because the page is complex.\n"
                + self._separator()
            )

        # Decide output format based on whether PolicyHub is active
        if policy_text and policy_text.strip():
            output_format_block = self._build_json_output_format()
        else:
            output_format_block = self._build_legacy_output_format()

        sep = self._separator()

        prompt_text = f"""
You are an expert web automation assistant. Your job is to generate **exactly one atomic UI action** at a time that will help complete the user's task on the current webpage. You should think like a human interacting with the page: observing, reasoning, and planning one small step at a time.

Do NOT produce multi-step plans, lists, code, or explanations -- only one action.

{sep}
INPUTS (do not repeat in output):

Instruction (goal): {instruction}
Current website screenshot:
<|vision_start|><|image_pad|><|vision_end|>

History of actions already taken:
{history_text}
{policy_block}
{sep}
THE ACTION SPACE (allowed actions):

1) click -- Click a target element
2) type -- Enter text into an input field
3) select -- Choose an option from a dropdown or similar control
4) FINISH -- No more actions needed

Only use the three UI-action types above (click, type, select). Do NOT output `scroll` or any other action.

All actions must follow the exact, concise output format described below.

{sep}
RESPONSE RULES (IMPORTANT):

* Generate **exactly ONE next action** that moves toward the goal.
* Do NOT repeat an action already in history.
* Do NOT make up UI element text -- use what is visible.
* Do NOT hallucinate or invent actions unrelated to the visible screenshot.
* If the task is already complete or no further UI action is needed, output exactly `FINISH` (without quotes).
{output_format_block}
{sep}
COMPLETE TASK EXAMPLES (Given for understanding):

Task Goal: "rent a car in Brooklyn - Central, NY on from April 9 to April 15."
Given Textual plan for all steps (which is expected output of the Action Generation for single step at point):
1. [heading]  CAR -> CLICK
2. [combobox]  Enter pick up city, airport name, or airport code. -> TYPE: Brooklyn Central
3. [div]  Brooklyn - Central (New York), US -> CLICK
4. [textbox]  Pickup -> CLICK
5. [button]  Sunday, April 9, 2023 -> CLICK
6. [button]  Saturday, April 15, 2023 -> CLICK
7. [button]  Find cars button. -> CLICK
FINISH

INVALID OUTPUTS (don't generate these):
* [] (empty value)
* click: #input-button
* input: search:nth-child(2)
* multi-step lists (e.g., "1. click..., 2. input...")
* explanations or thoughts in output
* commands not executable as UI action
* HTML, CSS selectors, code, or element ids

{sep}
Now based on the instruction, the screenshot, and history, generate the **next single action** and nothing else.

"""
        return {
            "screenshot": website_screenshot,
            "prompt": prompt_text
        }

    # ------------------------------------------------------------------
    # Private helpers for prompt building
    # ------------------------------------------------------------------

    @staticmethod
    def _separator() -> str:
        """Return the horizontal rule separator used in prompts."""
        return "-" * 76

    @staticmethod
    def _build_json_output_format() -> str:
        """Build the structured JSON output format block (used when PolicyHub is active)."""
        return """
OUTPUT FORMAT (respond with ONLY this JSON object, nothing else):

{"action": "[element_type] ELEMENT_TEXT -> CLICK", "confidence": 75, "policy_risk": false, "hitl_reason": ""}

Field rules:
- "action": One of the dataset-style action formats below:
    - `[div/button/url/a/link/img/label] ELEMENT_TEXT -> CLICK`
    - `[input/textarea] ELEMENT_TEXT -> TYPE: <typed value>`
    - `[select/option] ELEMENT_TEXT -> SELECT`
    - `FINISH`
- "confidence": Integer 0-100. Your realistic certainty about element selection.
    Report honestly — do NOT reduce confidence just because a page has many elements.
    - 80-100: Clearly correct element vector, obvious action
    - 60-79:  Likely correct, minor ambiguity but reasonable choice
    - 40-59:  Uncertain, multiple plausible candidates
    - 0-39:   Guessing or cannot determine proper target
- "policy_risk": Boolean. true ONLY if the action involves:
    destructive ops, financial transactions, credentials, CAPTCHA/security, or account changes.
    false for normal navigation, reading, searching, link clicks, typing queries.
- "hitl_reason": When policy_risk is true, explain what risk you detected.
    Otherwise empty string "".

Examples:
{"action": "[link] Wikibooks -> CLICK", "confidence": 72, "policy_risk": false, "hitl_reason": ""}
{"action": "[button] Search -> CLICK", "confidence": 90, "policy_risk": false, "hitl_reason": ""}
{"action": "[input] Enter city -> TYPE: Brooklyn Central", "confidence": 80, "policy_risk": false, "hitl_reason": ""}
{"action": "[checkbox] I am not a robot -> CLICK", "confidence": 60, "policy_risk": true, "hitl_reason": "CAPTCHA security mechanism."}
{"action": "[button] Delete item -> CLICK", "confidence": 55, "policy_risk": true, "hitl_reason": "Destructive action — delete keyword."}
{"action": "FINISH", "confidence": 85, "policy_risk": false, "hitl_reason": ""}

IMPORTANT: Respond with ONLY the raw JSON object. No markdown, no code blocks, no explanations.
"""

    @staticmethod
    def _build_legacy_output_format() -> str:
        """Build the legacy one-line output format block (backward compatible, no PolicyHub)."""
        return """
OUTPUT FORMAT (one line only):
Follow one of these concise dataset-style formats (one line only):

- For click:   `[div/button/url/a/link/img/label] ELEMENT_TEXT -> CLICK`
- For type:    `[input/textarea] ELEMENT_TEXT -> TYPE: <typed value>`
- For select:  `[select/option] ELEMENT_TEXT -> SELECT`
- For finish:  `FINISH`

Examples (match Multimodal-Mind2Web style exactly):
[div/button/url/a/link/img/label]  CAR -> CLICK
[input/textarea]  Enter pick up city, airport name, or airport code. -> TYPE: Brooklyn Central
[div/button/url/a/link/img/label]   Pickup -> CLICK
[div/button/url/a/link/img/label]   Sunday, April 9, 2023 -> CLICK
[div/button/url/a/link/img/label]  Saturday, April 15, 2023 -> CLICK
[div/button/url/a/link/img/label]  Community -> CLICK

* ACTION_TYPE must be exactly one of: `click`, `type`, `select`, `FINISH`.
* ACTION_DETAIL must be concise (<= 40 characters when possible) and use visible UI text. Do NOT add extra commentary or multi-step lists.
"""
