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

from PIL import Image
from supabase import create_client, Client

from AutoWeb.src.config import (
    get_supabase_url,
    get_supabase_service_role_key,
    get_supabase_bucket,
)


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
        print(f"  ✓ Supabase storage helper ready (bucket: {self.bucket})")

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------
    def upload(
        self,
        pil_image: Image.Image,
        filename: Optional[str] = None,
        quality: int = 85,
        folder: str = "autoweb_temp",
    ) -> str:
        """Upload a PIL Image and return its public URL.

        Args:
            pil_image:  PIL Image to upload.
            filename:   Optional custom filename. Auto-generated if omitted.
            quality:    JPEG quality (1-100).
            folder:     Sub-folder inside the bucket.

        Returns:
            Public URL string for the uploaded image.
        """
        if filename is None:
            filename = f"{uuid.uuid4().hex[:12]}_{int(time.time())}.jpg"

        path = f"{folder}/{filename}" if folder else filename

        # Convert PIL → JPEG bytes
        buf = io.BytesIO()
        rgb_img = pil_image.convert("RGB") if pil_image.mode != "RGB" else pil_image
        rgb_img.save(buf, format="JPEG", quality=quality, optimize=True)
        file_bytes = buf.getvalue()

        # Upload (upsert so re-runs don't fail)
        self.client.storage.from_(self.bucket).upload(
            path,
            file_bytes,
            file_options={"content-type": "image/jpeg", "upsert": "true"},
        )

        # Build public URL
        public_url = self.client.storage.from_(self.bucket).get_public_url(path)

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
            print(f"  ⚠ Supabase delete failed for {path}: {e}")
            return False
