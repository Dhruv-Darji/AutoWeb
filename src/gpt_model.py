"""
GPT-4o Vision Model Interface

Drop-in replacement for VLModel that uses the OpenAI GPT-4o API with
image-URL input (via Supabase temporary storage).

The class exposes the same `infer()` signature so that SeeActActionGenerator
and SeeActActionGrounding can use it without code changes.

Usage:
    model = GPTVisionModel()          # reads OPENAI_API_KEY, OPENAI_MODEL from .env
    result = model.infer(
        image_or_tensor=pil_image,    # PIL.Image — auto-uploaded to Supabase
        prompt_text="...",
        max_new_tokens=256,
    )
    print(result["output_text"])
"""

import time
from typing import Any, Dict, Optional, Union

from PIL import Image
from openai import OpenAI

from AutoWeb.src.config import get_openai_api_key, get_openai_model
from AutoWeb.src.utils.supabase_image import SupabaseImageHelper


class GPTVisionModel:
    """OpenAI GPT-4o vision model with Supabase image hosting."""

    def __init__(self):
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in .env")

        self.client = OpenAI(api_key=api_key)
        self.model_name = get_openai_model()
        self.supabase = SupabaseImageHelper()

        print(f"  ✓ GPTVisionModel ready (model: {self.model_name})")

    # ------------------------------------------------------------------
    # Public API — matches VLModel.infer() signature
    # ------------------------------------------------------------------
    def infer(
        self,
        image_or_tensor: Optional[Union[Image.Image, Any]] = None,
        prompt_text: str = "",
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.0,
        **kwargs,
    ) -> Dict:
        """Run GPT-4o inference with an optional image.

        Args:
            image_or_tensor: PIL Image (uploaded to Supabase) or None for text-only.
            prompt_text:     The prompt / system+user message.
            max_new_tokens:  Maps to OpenAI ``max_tokens``.
            do_sample:       Ignored (OpenAI uses temperature).
            temperature:     Sampling temperature (0 = deterministic).

        Returns:
            Dict with keys: output_text, raw_text, latency.
        """
        start = time.time()
        uploaded_filename: Optional[str] = None
        image_url: Optional[str] = None

        # --- Strip Qwen-specific vision placeholders from prompt ---
        clean_prompt = self._strip_qwen_placeholders(prompt_text)

        # --- Upload image if provided ---
        if image_or_tensor is not None and isinstance(image_or_tensor, Image.Image):
            import uuid
            uploaded_filename = f"gpt_{uuid.uuid4().hex[:10]}.jpg"
            image_url = self.supabase.upload(image_or_tensor, filename=uploaded_filename)

        # --- Build messages ---
        messages = self._build_messages(clean_prompt, image_url)

        # --- Call OpenAI ---
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            output_text = response.choices[0].message.content.strip()
        except Exception as e:
            output_text = ""
            print(f"    ✗ GPT-4o API call failed: {e}")
        finally:
            # --- Cleanup: delete image from Supabase immediately ---
            if uploaded_filename:
                self.supabase.delete(filename=uploaded_filename)

        latency = time.time() - start
        return {
            "output_text": output_text,
            "raw_text": output_text,
            "latency": latency,
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
    def _build_messages(prompt: str, image_url: Optional[str] = None):
        """Build OpenAI chat messages with optional image content."""
        content_parts = []

        if image_url:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": image_url, "detail": "high"},
            })

        content_parts.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content_parts}]
