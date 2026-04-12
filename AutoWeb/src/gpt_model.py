"""
GPT-4o Vision Model Interface

Drop-in replacement for VLModel that uses the OpenAI GPT-4o API with
image-URL input (via Supabase temporary storage).

The class exposes the same `infer()` signature so that SeeActActionGenerator
and SeeActActionGrounding can use it without code changes.

Tracks token usage and cost per call, per step, and per task.

Usage:
    model = GPTVisionModel()          # reads OPENAI_API_KEY, OPENAI_MODEL from .env
    result = model.infer(
        image_or_tensor=pil_image,    # PIL.Image — auto-uploaded to Supabase
        prompt_text="...",
        max_new_tokens=256,
    )
    print(result["output_text"])
    print(result["cost"])             # $ cost for this single call
    print(model.total_cost)           # cumulative $ cost across all calls
"""

import time
from typing import Any, Dict, Optional, Union

from PIL import Image
from openai import OpenAI

from AutoWeb.src.config import get_openai_api_key, get_openai_model
from AutoWeb.src.utils.supabase_image import SupabaseImageHelper
from AutoWeb.src.logger import logger


# ── Pricing per 1M tokens (USD) as of Feb 2025 ──────────────────────
# Source: https://openai.com/api/pricing/
# Update these if OpenAI changes pricing.
_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o": {
        "input":  2.50,   # $2.50 / 1M input tokens
        "output": 10.00,  # $10.00 / 1M output tokens
    },
    "gpt-4o-mini": {
        "input":  0.15,   # $0.15 / 1M input tokens
        "output": 0.60,   # $0.60 / 1M output tokens
    },
}
# Fallback for unknown models — use gpt-4o-mini pricing (safe upper bound)
_DEFAULT_PRICING = _PRICING["gpt-4o-mini"]


def _resolve_pricing_key(model_name: str) -> str:
    """Resolve pricing profile key from model name."""
    normalized = (model_name or "").lower()
    # Try exact key or prefix key first.
    if normalized in _PRICING:
        return normalized
    for key in sorted(_PRICING.keys(), key=len, reverse=True):
        if normalized.startswith(key):
            return key
    # Fallback: contains-match (handles model version suffixes and provider tags)
    for key in sorted(_PRICING.keys(), key=len, reverse=True):
        if key in normalized:
            return key
    return "gpt-4o-mini"


def _calc_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate USD cost for a single API call."""
    pricing_key = _resolve_pricing_key(model_name)
    pricing = _PRICING.get(pricing_key, _DEFAULT_PRICING)
    cost = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000
    return cost


class GPTVisionModel:
    """OpenAI GPT-4o vision model with Supabase image hosting and cost tracking."""

    def __init__(self):
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in .env")

        self.client = OpenAI(api_key=api_key)
        self.model_name = get_openai_model()
        self.pricing_key = _resolve_pricing_key(self.model_name)
        self.supabase = SupabaseImageHelper()

        # ── Cumulative cost tracking ──
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_cost: float = 0.0      # USD
        self.call_count: int = 0

        # Per-step accumulator (reset between steps via reset_step_cost())
        self._step_prompt_tokens: int = 0
        self._step_completion_tokens: int = 0
        self._step_cost: float = 0.0
        self._step_calls: int = 0

        logger.info(
            f"✓ GPTVisionModel ready (model: {self.model_name}, pricing_profile: {self.pricing_key})"
        )

    # ------------------------------------------------------------------
    # Cost helpers
    # ------------------------------------------------------------------
    def reset_step_cost(self):
        """Call at the start of each pipeline step to reset per-step counters."""
        self._step_prompt_tokens = 0
        self._step_completion_tokens = 0
        self._step_cost = 0.0
        self._step_calls = 0

    def get_step_cost(self) -> Dict:
        """Return per-step cost summary."""
        return {
            "prompt_tokens": self._step_prompt_tokens,
            "completion_tokens": self._step_completion_tokens,
            "total_tokens": self._step_prompt_tokens + self._step_completion_tokens,
            "cost_usd": round(self._step_cost, 6),
            "calls": self._step_calls,
        }

    def get_total_cost(self) -> Dict:
        """Return cumulative cost summary across all calls."""
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "cost_usd": round(self.total_cost, 6),
            "calls": self.call_count,
            "model": self.model_name,
            "pricing_profile": self.pricing_key,
        }

    # ------------------------------------------------------------------
    # Public API — matches VLModel.infer() signature
    # ------------------------------------------------------------------
    def infer(
        self,
        image_or_tensor: Optional[Union[Image.Image, Any]] = None,
        prompt_text: str = "",
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.2,
        image_detail: str = "low",
        **kwargs,
    ) -> Dict:
        """Run GPT-4o inference with an optional image.

        Args:
            image_or_tensor: PIL Image (uploaded to Supabase) or None for text-only.
            prompt_text:     The prompt / system+user message.
            max_new_tokens:  Maps to OpenAI ``max_tokens``.
            do_sample:       Ignored (OpenAI uses temperature).
            temperature:     Sampling temperature (0 = deterministic).
            image_detail:    OpenAI vision detail level: ``"high"`` (~25K tokens),
                             ``"low"`` (85 tokens), or ``"auto"``.  Use ``"low"``
                             for tasks that don't need fine visual detail (e.g.
                             selecting from pre-ranked text candidates).

        Returns:
            Dict with keys: output_text, raw_text, latency,
            prompt_tokens, completion_tokens, cost.
        """
        start = time.time()
        uploaded_filename: Optional[str] = None
        image_url: Optional[str] = None
        prompt_tokens = 0
        completion_tokens = 0
        call_cost = 0.0

        # --- Strip Qwen-specific vision placeholders from prompt ---
        clean_prompt = self._strip_qwen_placeholders(prompt_text)

        # --- Upload image if provided ---
        if image_or_tensor is not None and isinstance(image_or_tensor, Image.Image):
            import uuid
            uploaded_filename = f"gpt_{uuid.uuid4().hex[:10]}.jpg"
            image_url = self.supabase.upload(image_or_tensor, filename=uploaded_filename)

        # --- Build messages ---
        messages = self._build_messages(clean_prompt, image_url, detail=image_detail)

        # --- Call OpenAI (with retry on empty response and transient image fetch issues) ---
        max_retries = 4
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                )
                # Safe extraction — content can be None (e.g. content filter)
                raw_content = response.choices[0].message.content
                output_text = (raw_content or "").strip()

                # Extract token usage
                usage = response.usage
                if usage:
                    # Use self.pricing_key (resolved once at init from self.model_name)
                    # rather than response.model to avoid mismatches (e.g. response
                    # may return a dated alias that resolves to a different tier).
                    prompt_tokens += usage.prompt_tokens or 0
                    completion_tokens += usage.completion_tokens or 0
                    _cost = _calc_cost(self.model_name, usage.prompt_tokens or 0, usage.completion_tokens or 0)
                    call_cost += _cost

                    # Log response model on first call for debugging
                    if self.call_count == 0:
                        resp_model = getattr(response, "model", "?")
                        logger.info(f"    ℹ OpenAI response model: {resp_model} (pricing: {self.pricing_key})")

                    # Accumulate
                    self.total_prompt_tokens += usage.prompt_tokens or 0
                    self.total_completion_tokens += usage.completion_tokens or 0
                    self.total_cost += _cost
                    self.call_count += 1

                    self._step_prompt_tokens += usage.prompt_tokens or 0
                    self._step_completion_tokens += usage.completion_tokens or 0
                    self._step_cost += _cost
                    self._step_calls += 1

                # Retry if response is empty (sometimes GPT returns null content)
                if not output_text and attempt < max_retries:
                    logger.warning(f"⚠ GPT returned empty response (attempt {attempt}/{max_retries}), retrying...")
                    continue

                if not output_text:
                    finish_reason = response.choices[0].finish_reason if response.choices else "unknown"
                    logger.warning(f"⚠ GPT returned empty content after {attempt} attempts (finish_reason={finish_reason}, content_was_none={raw_content is None})")

                break  # success — exit retry loop

            except Exception as e:
                output_text = ""
                err_str = str(e)
                is_invalid_image_url = (
                    "invalid_image_url" in err_str.lower()
                    or "timeout while downloading" in err_str.lower()
                )

                logger.error(f"✗ GPT-4o API call failed (attempt {attempt}/{max_retries}): {e}")

                if is_invalid_image_url and image_url:
                    # Re-check URL readiness and apply stronger backoff for CDN propagation.
                    try:
                        self.supabase.wait_until_public(
                            image_url, timeout_s=20.0, interval_s=0.75
                        )
                    except Exception:
                        pass

                    if attempt < max_retries:
                        backoff_s = min(2 * attempt, 8)
                        logger.warning(f"⚠ image URL not ready for OpenAI fetch, retrying in {backoff_s}s...")
                        time.sleep(backoff_s)
                        continue

                if attempt >= max_retries:
                    break

        # --- Cleanup: delete image from Supabase immediately ---
        if uploaded_filename:
            self.supabase.delete(filename=uploaded_filename)

        latency = time.time() - start
        return {
            "output_text": output_text,
            "raw_text": output_text,
            "latency": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": round(call_cost, 6),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_qwen_placeholders(text: str) -> str:
        """Remove Qwen-specific ``<|vision_start|>...<|vision_end|>`` blocks."""
        import re
        # Remove entire vision block
        text = re.sub(
            r"<\|vision_start\|>.*?<\|vision_end\|>",
            "[see attached screenshot]",
            text,
            flags=re.DOTALL,
        )
        # Remove stray special tokens
        for tok in ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>"):
            text = text.replace(tok, "")
        return text.strip()

    @staticmethod
    def _build_messages(prompt: str, image_url: Optional[str] = None, detail: str = "low"):
        """Build OpenAI chat messages with optional image content.

        Args:
            detail: ``"high"`` (~25K tokens per image, fine detail),
                    ``"low"``  (85 tokens, 512×512 thumbnail), or
                    ``"auto"`` (let OpenAI decide).
        """
        content_parts = []

        if image_url:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": image_url, "detail": detail},
            })

        content_parts.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content_parts}]
