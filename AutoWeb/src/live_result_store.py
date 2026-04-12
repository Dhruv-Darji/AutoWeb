"""
Live Result Store — Persistent storage for live-mode SeeAct pipeline results.

Saves per-step artefacts (screenshot, prediction JSON) and a run-level
summary so results can be referenced when writing papers.

Directory layout produced::

    liveSiteResults/
        run_20260219_153012/
            run_summary.json          ← aggregate stats for the entire run
            Amazon/
                step_001/
                    screenshot.png    ← viewport capture at prediction time
                    result.json       ← full prediction result (plan, confidence, HITL, grounding)
                step_002/
                    ...
            Google/
                step_001/
                    ...
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from AutoWeb.src.logger import logger


class LiveResultStore:
    """
    Handles persisting screenshots and prediction results for every
    live-mode step, plus a run-level summary JSON.

    Usage::

        store = LiveResultStore(base_dir="liveSiteResults")
        store.save_step(site_name="Amazon", step_num=1,
                        screenshot=pil_img, result=result_dict)
        ...
        store.save_run_summary(all_results)
    """

    def __init__(self, base_dir: str = "liveSiteResults"):
        """
        Create a timestamped run directory under *base_dir*.

        Args:
            base_dir: Root folder for all live results.
        """
        self.base_dir = Path(base_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[LiveResultStore] Run directory: {self.run_dir}")

    # ------------------------------------------------------------------
    # Per-step persistence
    # ------------------------------------------------------------------

    def save_step(
        self,
        site_name: str,
        step_num: int,
        screenshot: Optional[Image.Image],
        result: Dict,
    ) -> Path:
        """
        Persist a single step's screenshot and prediction result.

        Args:
            site_name:  Human-readable site name (used as folder name).
            step_num:   1-based step index within the site.
            screenshot: PIL.Image of the viewport (saved as PNG).
            result:     Full prediction result dict from ``predict_live_step()``.

        Returns:
            Path to the step directory.
        """
        # Sanitise site name for filesystem
        safe_name = re.sub(r'[<>:"/\\|?*]', "_", site_name).strip()
        step_dir = self.run_dir / safe_name / f"step_{step_num:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # 1. Screenshot
        if screenshot is not None:
            screenshot_path = step_dir / "screenshot.png"
            screenshot.save(str(screenshot_path), format="PNG")
            logger.debug(f"  Saved screenshot → {screenshot_path}")

        # 2. Result JSON (make a serialisable copy)
        result_path = step_dir / "result.json"
        serialisable = self._make_serialisable(result)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2, ensure_ascii=False)
        logger.debug(f"  Saved result     → {result_path}")

        return step_dir

    # ------------------------------------------------------------------
    # Per-site summary (saved after each site completes)
    # ------------------------------------------------------------------

    def save_site_summary(
        self,
        site_name: str,
        site_results: List[Dict],
    ) -> Path:
        """
        Write a ``site_summary.json`` inside the site folder after all steps
        for that site are finished.  Called incrementally so results survive
        a crash on a later site.

        Args:
            site_name:    Human-readable site name.
            site_results: List of result dicts for *this* site only.

        Returns:
            Path to the site summary JSON.
        """
        safe_name = re.sub(r'[<>:"/\\|?*]', "_", site_name).strip()
        site_dir = self.run_dir / safe_name
        site_dir.mkdir(parents=True, exist_ok=True)

        total = len(site_results)
        successful = sum(1 for r in site_results if r.get("success"))
        hitl_count = sum(1 for r in site_results if r.get("hitl_triggered"))
        avg_conf = (
            sum(r.get("composite_confidence", 0) for r in site_results) / total
            if total else 0
        )

        summary = {
            "site_name": site_name,
            "url": site_results[0].get("url", "") if site_results else "",
            "timestamp": datetime.now().isoformat(),
            "total_steps": total,
            "successful": successful,
            "hitl_triggers": hitl_count,
            "hitl_trigger_rate": round(hitl_count / total, 4) if total else 0,
            "avg_confidence": round(avg_conf, 2),
            "steps": [
                {
                    "step_num": r.get("step_num", 0),
                    "instruction": r.get("instruction", ""),
                    "output_plan": r.get("output_plan", ""),
                    "llm_confidence": r.get("llm_confidence", 0),
                    "composite_confidence": r.get("composite_confidence", 0),
                    "hitl_triggered": r.get("hitl_triggered", False),
                    "hitl_reason": r.get("hitl_reason", ""),
                    "latency": round(r.get("latency", 0), 3),
                }
                for r in site_results
            ],
        }

        summary_path = site_dir / "site_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(
            f"[LiveResultStore] Site summary saved → {summary_path}  "
            f"({total} steps, {hitl_count} HITL)"
        )
        return summary_path

    # ------------------------------------------------------------------
    # Run-level summary
    # ------------------------------------------------------------------

    def save_run_summary(self, all_results: List[Dict]) -> Path:
        """
        Write a ``run_summary.json`` with aggregate stats and per-step refs.

        Args:
            all_results: List of all result dicts accumulated during the run.

        Returns:
            Path to the summary JSON.
        """
        total = len(all_results)
        successful = sum(1 for r in all_results if r.get("success"))
        hitl_count = sum(1 for r in all_results if r.get("hitl_triggered"))
        avg_conf = (
            sum(r.get("composite_confidence", 0) for r in all_results) / total
            if total else 0
        )
        total_latency = sum(r.get("latency", 0) for r in all_results)

        # Per-site breakdown
        sites_breakdown = {}
        for r in all_results:
            s = r.get("site", "unknown")
            sites_breakdown.setdefault(s, []).append(r)

        per_site = {}
        for s, steps in sites_breakdown.items():
            per_site[s] = {
                "total_steps": len(steps),
                "successful": sum(1 for r in steps if r.get("success")),
                "hitl_triggers": sum(1 for r in steps if r.get("hitl_triggered")),
                "avg_confidence": round(
                    sum(r.get("composite_confidence", 0) for r in steps) / len(steps), 2
                ),
                "steps": [
                    {
                        "step_num": r.get("step_num", 0),
                        "instruction": r.get("instruction", ""),
                        "url": r.get("url", ""),
                        "output_plan": r.get("output_plan", ""),
                        "llm_confidence": r.get("llm_confidence", 0),
                        "composite_confidence": r.get("composite_confidence", 0),
                        "hitl_triggered": r.get("hitl_triggered", False),
                        "hitl_reason": r.get("hitl_reason", ""),
                        "grounding_success": (r.get("grounding_result") or {}).get("success", False),
                        "latency": round(r.get("latency", 0), 3),
                        "cost": r.get("cost", {}),
                    }
                    for r in steps
                ],
            }

        summary = {
            "run_dir": str(self.run_dir),
            "timestamp": datetime.now().isoformat(),
            "aggregate": {
                "total_steps": total,
                "successful": successful,
                "hitl_triggers": hitl_count,
                "hitl_trigger_rate": round(hitl_count / total, 4) if total else 0,
                "avg_confidence": round(avg_conf, 2),
                "total_latency_s": round(total_latency, 2),
            },
            "per_site": per_site,
        }

        summary_path = self.run_dir / "run_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"[LiveResultStore] Run summary saved → {summary_path}")
        print(f"\n  📁 Results saved to: {self.run_dir}")
        print(f"     Summary: {summary_path}")
        return summary_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_serialisable(obj):
        """
        Recursively convert a result dict into a JSON-safe structure.

        Handles PIL Images, sets, bytes, and other non-serialisable types.
        """
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {k: LiveResultStore._make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [LiveResultStore._make_serialisable(v) for v in obj]
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, bytes):
            return f"<bytes len={len(obj)}>"
        if isinstance(obj, Image.Image):
            return f"<PIL.Image {obj.size[0]}x{obj.size[1]} mode={obj.mode}>"
        if isinstance(obj, Path):
            return str(obj)
        # Fallback: string representation
        try:
            return str(obj)
        except Exception:
            return "<non-serialisable>"
