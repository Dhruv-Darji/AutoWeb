"""
Browser Driver — Playwright-based browser automation for live SeeAct pipeline.

Provides:
    - Navigate to a URL
    - Capture screenshot as PIL.Image
    - Extract page HTML (full DOM) for action grounding
    - Close browser resources
"""

import io
from typing import Dict, Optional

from PIL import Image

from AutoWeb.src.logger import logger


class BrowserDriver:
    """
    Headful Chromium browser driven by Playwright (sync API).

    Usage::

        driver = BrowserDriver(headless=False)
        driver.navigate("https://www.amazon.com")
        state = driver.capture_state()
        # state == {"screenshot": PIL.Image, "cleaned_html": str, "url": str}
        driver.close()
    """

    def __init__(self, headless: bool = False, viewport_width: int = 1280, viewport_height: int = 720):
        """
        Args:
            headless:        Run browser in headless mode (False for visual debugging).
            viewport_width:  Browser viewport width in pixels.
            viewport_height: Browser viewport height in pixels.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError(
                "Playwright is required for live mode.\n"
                "  pip install playwright\n"
                "  playwright install chromium"
            )

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=headless)
        self._context = self._browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height},
            # Accept common locale/permissions for testing
            locale="en-US",
        )
        self._page = self._context.new_page()
        self._page.set_default_timeout(30_000)  # 30 s

        logger.info(
            f"[BrowserDriver] Chromium launched  headless={headless}  "
            f"viewport={viewport_width}x{viewport_height}"
        )

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def navigate(self, url: str, wait_until: str = "domcontentloaded") -> None:
        """
        Navigate to *url* and wait for the page to be ready.

        Args:
            url:        Target URL.
            wait_until: Playwright wait strategy — ``"domcontentloaded"``
                        or ``"networkidle"`` (slower but more complete).
        """
        logger.info(f"[BrowserDriver] Navigating to {url} ...")
        self._page.goto(url, wait_until=wait_until)
        logger.info(f"[BrowserDriver] Page loaded: {self._page.title()}")

    # ------------------------------------------------------------------
    # State capture
    # ------------------------------------------------------------------

    def capture_state(self) -> Dict:
        """
        Capture the current page state for the SeeAct pipeline.

        Returns:
            Dict with:
                ``screenshot``    — PIL.Image (RGB) of the current viewport
                ``cleaned_html``  — full outer-HTML of ``<body>`` (str)
                ``url``           — current page URL (str)
                ``title``         — page title (str)
        """
        # 1. Full-page screenshot → PIL Image (matches Mind2Web end-to-end style)
        png_bytes = self._page.screenshot(type="png", full_page=True)
        screenshot = Image.open(io.BytesIO(png_bytes)).convert("RGB")

        # 2. Extract full body HTML (used by action_grounding to find elements)
        cleaned_html = self._page.evaluate("() => document.body.outerHTML")

        return {
            "screenshot": screenshot,
            "cleaned_html": cleaned_html or "",
            "url": self._page.url,
            "title": self._page.title(),
        }

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    @property
    def current_url(self) -> str:
        return self._page.url

    @property
    def page(self):
        """Expose raw Playwright Page for advanced usage."""
        return self._page

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close browser and Playwright resources."""
        try:
            self._context.close()
            self._browser.close()
            self._pw.stop()
            logger.info("[BrowserDriver] Browser closed.")
        except Exception as e:
            logger.warning(f"[BrowserDriver] Error during close: {e}")
