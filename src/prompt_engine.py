import json
from typing import List, Dict, Optional, Tuple


"""
CONSTANT PROMPTS USED IN THE APPLICATION
"""

DEFAULT_STRICT_TEMPLATE = """You are a UI action predictor for web automation.

Given a webpage screenshot and a task instruction, predict the next single UI action.

Output EXACTLY ONE JSON object with NO additional text. Use this exact format:

{
  "action_type": "click|type|scroll|noop",
  "target": {"selector": "CSS selector"} OR {"coords": [x, y]} OR {"bbox": [x, y, w, h]} OR null,
  "value": "text to type" OR null,
  "confidence": 0.0 to 1.0
}

Rules:
- action_type: MUST be one of: click, type, scroll, noop
- target: For click/type, provide selector OR coords OR bbox. For scroll/noop, use null
- value: ONLY for type actions. Otherwise null
- confidence: Your confidence score (0.0 = no confidence, 1.0 = certain)

Output ONLY the JSON. No explanations. No markdown. No extra text."""

FEW_SHOT_EXAMPLE_PROMPT = """
### Example
Screenshot: <image>
Task: "{task}"
Output:
{json}
"""

class PromptEngine:
    """
    Build prompts for Qwen2-VL-2B-Instruct.

    Usage:
        engine = PromptEngine()
        prompt_text, attached_images = engine.build_prompt(
            task_text="Click Login",
            dom_snippet=...,                 # optional
            examples=[{...}, ...],           # optional few-shot examples
            variant="strict_json"            # or "few_shot", "dom_augmented"
        )
    """
    def __init__(self,
                 strict_template: str = DEFAULT_STRICT_TEMPLATE,
                 few_shot_example_template: str = FEW_SHOT_EXAMPLE_PROMPT,
                 max_dom_chars: int = 2000):
        
        self.strict_template = strict_template.strip()
        self.example_template = few_shot_example_template.strip()
        self.max_dom_chars = max_dom_chars

    def _format_example(self, example: Dict) -> str:
        """
        example expected keys:
          - 'task' : str
          - 'json' : dict (oracle JSON)
          - 'image_key' : optional placeholder like '<image>'
        """
        task = self._clean_text(example.get("task", ""))
        example_json = example.get("json", {})
        # ensure compact JSON string (no spaces/newlines) to keep prompt short
        json_str = json.dumps(example_json, ensure_ascii=False)
        image_placeholder = example.get("image_key", "<image>")
        s = self.example_template.replace("<image>", image_placeholder)
        s = s.replace("{task}", task).replace("{json}", json_str)
        return s

    def _format_dom_elements(self, dom_elements: List[Dict], max_elements: int = 50) -> str:
        """
        Format preprocessor's dom_elements into a compact string for the prompt.
        Returns: string like "1. <button id='login'> Login\n2. <input name='email'>\n..."
        """
        if not dom_elements:
            return ""
        
        lines = []
        for idx, elem in enumerate(dom_elements[:max_elements], 1):
            tag = elem.get("tag", "unknown")
            text = elem.get("text", "")[:50]  # limit text length
            eid = elem.get("id", "")
            clazz = elem.get("class", "")
            
            # format: "1. <button id='login' class='btn'> Login"
            attrs_str = ""
            if eid:
                attrs_str += f" id='{eid}'"
            if clazz:
                attrs_str += f" class='{clazz[:30]}'"
            
            line = f"{idx}. <{tag}{attrs_str}>"
            if text:
                line += f" {text}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def _trim_dom(self, dom_snippet: Optional[str]) -> str:
        if not dom_snippet:
            return ""
        
        s = self._clean_text(dom_snippet)
        if len(s) <= self.max_dom_chars:
            return s
        
        head = s[: int(self.max_dom_chars * 0.6)]
        tail = s[-int(self.max_dom_chars * 0.4):]
        return head + "\n...\n" + tail

    def _clean_text(self, s: str) -> str:
        if s is None:
            return ""
        s = " ".join(s.split())
        return s

    def build_prompt(
            self,
            task_text: str,
            dom_snippet: Optional[str] = None,
            dom_elements: Optional[List[Dict]] = None,
            examples: Optional[List[Dict]] = None,
            variant: str = "strict_json",
            include_dom: bool = True,
            image_key: str = "<image>"
            ) -> Tuple[str, List[str]]:
        """
        Build a prompt string and return (prompt_text, attached_image_keys).


        Args:
            task_text: natural-language instruction to perform on the page.
            dom_snippet: optional HTML string or short DOM text to include (raw HTML).
            dom_elements: optional structured DOM elements from preprocessor (preferred over raw HTML).
            examples: optional list of few-shot examples; each example is a dict:
                      { "task": str, "json": dict, "image_key": "<image>" }
            variant: one of "strict_json", "few_shot", "dom_augmented"
            include_dom: whether to include DOM context in prompt if provided
            image_key: placeholder string (e.g., "<image>") used in prompt; actual PIL image is passed to model separately

        Returns:
            (prompt_text, [image_key])  -- prompt text and list of image placeholder keys
        """

        task_text_clean = self._clean_text(task_text)
        
        # Prefer structured dom_elements over raw HTML
        dom_context = None
        if include_dom:
            if dom_elements:
                dom_context = self._format_dom_elements(dom_elements)
            elif dom_snippet:
                dom_context = self._trim_dom(dom_snippet)

        parts: List[str] = []
        attached_images = [image_key]

        # Base instruction (strict) always present to enforce JSON-only output
        parts.append(self.strict_template)

        # add optional DOM context
        if dom_context:
            parts.append("Context (DOM elements):")
            parts.append(dom_context)

        # few-shot
        if variant in ("few_shot", "dom_augmented") and examples:
            # format up to 3 examples by default
            ex_list = examples[:3]
            formatted = "\n\n".join([self._format_example(e) for e in ex_list])
            parts.append("Follow these examples:")
            parts.append(formatted)
        
        # the query / task
        parts.append(f"Screenshot: {image_key}")
        parts.append(f"Task: \"{task_text_clean}\"")
        parts.append("Output:")

        prompt = "\n\n".join(parts)
        return prompt, attached_images

    @staticmethod
    def make_example(task: str, input_json: Dict, image_key: str = "<image>") -> Dict:
        return {"task": task, "json": input_json, "image_key": image_key}