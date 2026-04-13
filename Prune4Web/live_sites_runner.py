"""Run Prune4Web on AutoWeb live_sites.json and persist JSON + .log outputs."""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from Prune4Web.run_prune4web import (
    OpenAI,
    StepResult,
    USAGE_TRACKER,
    print_step_results,
    run_prune4web_live,
)
import Prune4Web.run_prune4web as _rpw


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "live_sites"
DEFAULT_SITES_PATH = _PROJECT_ROOT / "AutoWeb" / "src" / "live_sites.json"


def _safe_name(name: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*]+', "_", name).strip()
    return safe or "site"


def _load_sites(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("sites", [])


def _site_summary(site: Dict, steps: List[StepResult], elapsed: float, error: str = "") -> Dict:
    total = len(steps)
    executed = sum(1 for s in steps if s.executed)
    hitl_count = sum(1 for s in steps if s.hitl_triggered)
    return {
        "name": site.get("name", ""),
        "url": site.get("url", ""),
        "domain": site.get("domain", ""),
        "instruction": site.get("default_instruction", ""),
        "risk_level": site.get("risk_level", ""),
        "requires_hitl": site.get("requires_hitl", False),
        "policy_category": site.get("policy_category", ""),
        "elapsed_seconds": round(elapsed, 2),
        "error": error,
        "total_steps": total,
        "steps_executed": executed,
        "steps_failed": total - executed,
        "hitl_triggers": hitl_count,
        "hitl_trigger_rate": round(hitl_count / total, 4) if total else 0.0,
        "steps": [
            {
                "step": s.step_idx,
                "sub_task": s.sub_task,
                "action": s.action_type,
                "action_value": s.action_value,
                "llm_confidence": s.llm_confidence,
                "composite_confidence": s.composite_confidence,
                "policy_risk_flag": s.policy_risk_flag,
                "hitl_triggered": s.hitl_triggered,
                "hitl_reason": s.hitl_reason,
                "keyword_weights": s.keyword_weights,
                "dom_nodes": s.num_dom_nodes,
                "candidates": s.num_candidates,
                "grounded_uid": s.grounded_uid,
                "grounded_element": s.grounded_element_summary,
                "grounder_confidence": s.confidence,
                "grounder_reasoning": s.reasoning,
                "executed": s.executed,
                "executed_selector": s.executed_selector,
                "grounding_error": s.grounding_error,
            }
            for s in steps
        ],
    }


def _append_site_log(log_path: Path, index: int, total: int, site: Dict, summary: Dict) -> None:
    lines = [
        f"[{index}/{total}] {summary['name']}",
        f"  url={summary['url']}",
        f"  domain={summary['domain']}",
        f"  instruction={summary['instruction']}",
        f"  risk_level={summary['risk_level']} requires_hitl={summary['requires_hitl']}",
        f"  elapsed={summary['elapsed_seconds']}s error={summary['error'] or '-'}",
        (
            f"  steps={summary['total_steps']} executed={summary['steps_executed']} "
            f"hitl={summary['hitl_triggers']}"
        ),
    ]
    for s in summary["steps"]:
        lines.append(
            f"    step={s['step']} action={s['action']} executed={s['executed']} "
            f"hitl={s['hitl_triggered']} llm_conf={s['llm_confidence']:.1f} "
            f"comp_conf={s['composite_confidence']:.1f}"
        )
        if s["hitl_reason"]:
            lines.append(f"      hitl_reason={s['hitl_reason']}")
        if s["grounding_error"]:
            lines.append(f"      grounding_error={s['grounding_error']}")
    lines.append("")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    p = argparse.ArgumentParser(description="Run Prune4Web on AutoWeb live_sites.json")
    p.add_argument("--sites-path", default=str(DEFAULT_SITES_PATH), help="Path to live_sites.json")
    p.add_argument("--max-steps", type=int, default=1, help="Max steps per site")
    p.add_argument("--headless", action="store_true", help="Run browser headless")
    p.add_argument("--limit", type=int, default=0, help="Limit number of sites (0 = all)")
    p.add_argument("--site-filter", nargs="+", default=[], help="Run sites matching any substring")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--use-local-grounder", dest="use_local_grounder", action="store_true")
    group.add_argument("--no-use-local-grounder", dest="use_local_grounder", action="store_false")
    p.set_defaults(use_local_grounder=True)
    args = p.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in .env")
    _rpw._CLIENT = OpenAI(api_key=api_key)

    if args.use_local_grounder:
        os.environ["GROUNDER_USE_LOCAL"] = "1"
    else:
        os.environ.pop("GROUNDER_USE_LOCAL", None)

    sites_path = Path(args.sites_path).resolve()
    if not sites_path.exists():
        raise FileNotFoundError(f"Sites file not found: {sites_path}")

    sites = _load_sites(sites_path)
    if args.site_filter:
        needles = [s.lower() for s in args.site_filter]
        sites = [s for s in sites if any(n in s.get("name", "").lower() for n in needles)]
    if args.limit > 0:
        sites = sites[: args.limit]
    if not sites:
        print("No sites selected.")
        return

    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Prune4Web Live Sites Run\n")
        f.write(f"timestamp={run_ts}\n")
        f.write(f"sites_path={sites_path}\n")
        f.write(f"max_steps={args.max_steps}\n")
        f.write(f"headless={args.headless}\n")
        f.write(f"use_local_grounder={args.use_local_grounder}\n\n")

    print("=" * 72)
    print("PRUNE4WEB LIVE SITES RUNNER")
    print("=" * 72)
    print(f"  Sites file : {sites_path}")
    print(f"  Total sites: {len(sites)}")
    print(f"  Max steps  : {args.max_steps}")
    print(f"  Output dir : {run_dir}")
    print("=" * 72)

    summaries: List[Dict] = []
    started = time.time()

    for idx, site in enumerate(sites, start=1):
        name = site.get("name", site.get("url", f"site_{idx}"))
        url = site.get("url", "")
        instruction = site.get("default_instruction", "").strip()
        domain = site.get("domain", "").strip()

        if not url or not instruction:
            summary = _site_summary(site, [], elapsed=0.0, error="missing_url_or_instruction")
            summaries.append(summary)
            _append_site_log(log_path, idx, len(sites), site, summary)
            continue

        print(f"\n[{idx}/{len(sites)}] {name}")
        print(f"  URL         : {url}")
        print(f"  Instruction : {instruction}")
        print(f"  Domain      : {domain or '-'}")

        steps: List[StepResult] = []
        error = ""
        t0 = time.time()
        try:
            steps = asyncio.run(
                run_prune4web_live(
                    url=url,
                    task=instruction,
                    max_steps=args.max_steps,
                    headless=args.headless,
                    website_domain=domain,
                )
            )
            print_step_results(steps)
        except KeyboardInterrupt:
            error = "interrupted"
        except Exception as exc:
            error = str(exc)[:200]
        elapsed = time.time() - t0

        summary = _site_summary(site, steps, elapsed=elapsed, error=error)
        summaries.append(summary)

        site_file = run_dir / f"{_safe_name(name)}.json"
        with open(site_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        _append_site_log(log_path, idx, len(sites), site, summary)
        print(f"  Saved       : {site_file}")

        if error == "interrupted":
            break

    total_elapsed = time.time() - started
    total_steps = sum(s["total_steps"] for s in summaries)
    total_executed = sum(s["steps_executed"] for s in summaries)
    total_hitl = sum(s["hitl_triggers"] for s in summaries)
    total_cost = sum(c.estimated_cost_usd or 0.0 for c in USAGE_TRACKER.calls)

    run_summary = {
        "timestamp": run_ts,
        "sites_path": str(sites_path),
        "output_dir": str(run_dir),
        "config": {
            "max_steps": args.max_steps,
            "headless": args.headless,
            "use_local_grounder": args.use_local_grounder,
        },
        "aggregate": {
            "total_sites": len(summaries),
            "total_steps": total_steps,
            "steps_executed": total_executed,
            "hitl_triggers": total_hitl,
            "hitl_trigger_rate": round(total_hitl / total_steps, 4) if total_steps else 0.0,
            "step_success_rate": round(total_executed / total_steps, 4) if total_steps else 0.0,
            "elapsed_seconds": round(total_elapsed, 2),
            "api_cost_usd": round(total_cost, 6),
        },
        "sites": [{k: v for k, v in s.items() if k != "steps"} for s in summaries],
    }

    summary_path = run_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\nRUN SUMMARY\n")
        f.write(json.dumps(run_summary["aggregate"], ensure_ascii=False) + "\n")
        f.write(f"summary_path={summary_path}\n")

    USAGE_TRACKER.print_summary()
    print(f"\nRun summary saved -> {summary_path}")
    print(f"Text log saved    -> {log_path}")


if __name__ == "__main__":
    main()

