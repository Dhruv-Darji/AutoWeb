"""
Supabase Storage Helper for temporary image hosting.

Uploads a PIL Image to a Supabase Storage bucket, returns the public URL,
and provides a cleanup method to delete it immediately after use.

Usage:
    helper = SupabaseImageHelper()
    url = helper.upload(pil_image, filename="step_001.jpg")
    # ... use url with GPT-4o ...
    helper.delete(filename="step_001.jpg")
"""

import io
import uuid
import time
from typing import Optional

import requests as _requests          # used for preflight image verification
from PIL import Image
from supabase import create_client, Client

from AutoWeb.src.config import (
    get_supabase_url,
    get_supabase_service_role_key,
    get_supabase_bucket,
)
from AutoWeb.src.logger import logger


class SupabaseImageHelper:
    """Upload / delete temporary images in Supabase Storage."""

    def __init__(self, bucket: Optional[str] = None):
        url = get_supabase_url()
        key = get_supabase_service_role_key()
        if not url or not key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env"
            )

        self.client: Client = create_client(url, key)
        self.bucket = bucket or get_supabase_bucket()
        logger.info(f"  ✓ Supabase storage helper ready (bucket: {self.bucket})")

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------
    def wait_until_public(
        self,
        public_url: str,
        timeout_s: float = 20.0,
        interval_s: float = 1.0,
    ) -> bool:
        """Poll public URL until the **image body** is actually downloadable.

        Uses a real HTTP GET with ``Range: bytes=0-1023`` so the CDN is
        forced to serve (and cache) the object body — not just headers.
        This prevents the race condition where OpenAI's servers hit a
        different CDN edge that hasn't replicated the object yet.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                resp = _requests.get(
                    public_url,
                    headers={"Range": "bytes=0-1023"},
                    timeout=6,
                    stream=True,      # don't buffer full image
                )
                # 200 = full body, 206 = partial content — both mean the
                # image body is available on this CDN edge.
                if resp.status_code in (200, 206):
                    logger.info(f"  ✓ Image URL is now publicly accessible and CDN-cached (status code: {resp.status_code})")
                    # Read the chunk to ensure it's real bytes, not an
                    # error page.
                    chunk = resp.content
                    # still keep delay of 2s between try:
                    time.sleep(2)
                    if len(chunk) > 0:
                        return True
                    else:
                        logger.warning(f"  ⚠ Image URL responded with empty body (status code: {resp.status_code}) — may not be fully cached yet: {public_url}")
                        raise ValueError("Empty body")
                resp.close()
            except Exception:
                pass
            time.sleep(interval_s)
        return False

    def upload(
        self,
        pil_image: Image.Image,
        filename: Optional[str] = None,
        folder: str = "autoweb_temp",
        max_dim: int = 2048,
    ) -> str:
        """Upload a PIL Image as JPEG and return its public URL.

        Large images (e.g. Mind2Web full-page screenshots at 1280×5429) are
        resized so the longest side fits within ``max_dim`` pixels.  This
        matches OpenAI's internal limit — GPT-4o scales images to 2048×2048
        anyway — and keeps file sizes small so the Supabase CDN can serve
        them before OpenAI's download timeout.

        Args:
            pil_image:  PIL Image to upload.
            filename:   Optional custom filename. Auto-generated if omitted.
            folder:     Sub-folder inside the bucket.
            max_dim:    Maximum width or height (default 2048).

        Returns:
            Public URL string for the uploaded image.
        """
        if filename is None:
            filename = f"{uuid.uuid4().hex[:12]}_{int(time.time())}.jpg"

        path = f"{folder}/{filename}" if folder else filename

        rgb_img = pil_image.convert("RGB") if pil_image.mode != "RGB" else pil_image

        # Resize if either dimension exceeds max_dim (keeps aspect ratio)
        w, h = rgb_img.size
        if w > max_dim or h > max_dim:
            scale = max_dim / max(w, h)
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            rgb_img = rgb_img.resize(new_size, Image.Resampling.LANCZOS)

        # Convert PIL → JPEG bytes (quality=85)
        buf = io.BytesIO()
        rgb_img.save(buf, format="JPEG", quality=85)
        file_bytes = buf.getvalue()

        # Upload (upsert so re-runs don't fail)
        self.client.storage.from_(self.bucket).upload(
            path,
            file_bytes,
            file_options={"content-type": "image/jpeg", "upsert": "true"},
        )

        # Build public URL
        public_url = self.client.storage.from_(self.bucket).get_public_url(path)

        # Wait until URL is actually downloadable (body bytes, not just headers)
        # before handing it to OpenAI — eliminates CDN propagation race.
        ready = self.wait_until_public(public_url, timeout_s=20.0, interval_s=1.0)
        if not ready:
            logger.warning(f"  ⚠ Image URL not confirmed downloadable within 20 s — OpenAI may still timeout: {public_url}")

        return public_url

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    def delete(self, filename: str, folder: str = "autoweb_temp") -> bool:
        """Delete a previously uploaded image.

        Args:
            filename: Filename used during upload.
            folder:   Sub-folder inside the bucket (must match upload).

        Returns:
            True if delete succeeded, False otherwise.
        """
        path = f"{folder}/{filename}" if folder else filename
        try:
            self.client.storage.from_(self.bucket).remove([path])
            return True
        except Exception as e:
            logger.warning(f"  ⚠ Supabase delete failed for {path}: {e}")
            return False
