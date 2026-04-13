"""
live_benchmark.py — Run Prune4Web on 30 live websites across 3 difficulty tiers.

Usage:
    "D:\Environments\ml-env\Scripts\python.exe" -X utf8 Prune4Web/live_benchmark.py
    "D:\Environments\ml-env\Scripts\python.exe" -X utf8 Prune4Web/live_benchmark.py --headless
    "D:\Environments\ml-env\Scripts\python.exe" -X utf8 Prune4Web/live_benchmark.py --use-local-grounder
    "D:\Environments\ml-env\Scripts\python.exe" -X utf8 Prune4Web/live_benchmark.py --categories easy medium

Each task gets up to --max-steps (default 5). Results are saved per-site and
as an aggregate JSON in Prune4Web/results/live_benchmark/.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

# ── Bootstrap ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from Prune4Web.run_prune4web import (
    run_prune4web_live,
    print_step_results,
    USAGE_TRACKER,
    StepResult,
)

# ── 30 live benchmark tasks across 3 categories ─────────────────────────
# Each entry: (name, url, task, category, max_steps)
#   easy   — single-step navigation, click a visible link/button
#   medium — 2-3 steps, may involve typing, selecting, or scrolling
#   hard   — 3-5 steps, multi-step flows, dynamic pages, forms

BENCHMARK_TASKS = [
    # ======== EASY (10 tasks) — single clear action ========
    {
        "name": "Wikipedia Search",
        "url": "https://www.wikipedia.org/",
        "task": "Search for 'Machine Learning' using the search box",
        "category": "easy",
        "max_steps": 3,
    },
    {
        "name": "DuckDuckGo Search",
        "url": "https://duckduckgo.com/",
        "task": "Search for 'web automation'",
        "category": "easy",
        "max_steps": 3,
    },
    {
        "name": "Python.org Downloads",
        "url": "https://www.python.org/",
        "task": "Click on the Downloads menu link",
        "category": "easy",
        "max_steps": 2,
    },
    {
        "name": "Hacker News Top Story",
        "url": "https://news.ycombinator.com/",
        "task": "Click on the first news article title on the page",
        "category": "easy",
        "max_steps": 2,
    },
    {
        "name": "GitHub Explore",
        "url": "https://github.com/explore",
        "task": "Click the Trending link in the navigation",
        "category": "easy",
        "max_steps": 2,
    },
    {
        "name": "BBC News",
        "url": "https://www.bbc.com/news",
        "task": "Click on the first headline article on the page",
        "category": "easy",
        "max_steps": 2,
    },
    {
        "name": "MDN Web Docs",
        "url": "https://developer.mozilla.org/en-US/",
        "task": "Search for 'flexbox' using the search bar",
        "category": "easy",
        "max_steps": 3,
    },
    {
        "name": "Stack Overflow",
        "url": "https://stackoverflow.com/",
        "task": "Click on the Questions link in the left sidebar",
        "category": "easy",
        "max_steps": 2,
    },
    {
        "name": "Reddit Homepage",
        "url": "https://www.reddit.com/",
        "task": "Click on the first post title visible on the page",
        "category": "easy",
        "max_steps": 2,
    },
    {
        "name": "W3Schools HTML",
        "url": "https://www.w3schools.com/html/default.asp",
        "task": "Click the 'Next' button to go to the next tutorial page",
        "category": "easy",
        "max_steps": 2,
    },

    # ======== MEDIUM (10 tasks) — 2-3 steps, typing or selecting ========
    {
        "name": "Wikipedia Article Nav",
        "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "task": "Scroll down and click the 'History' section link in the table of contents",
        "category": "medium",
        "max_steps": 4,
    },
    {
        "name": "Google Search",
        "url": "https://www.google.com/",
        "task": "Type 'Prune4Web web agent' in the search box and press search",
        "category": "medium",
        "max_steps": 3,
    },
    {
        "name": "YouTube Search",
        "url": "https://www.youtube.com/",
        "task": "Search for 'web automation tutorial' and click the first video result",
        "category": "medium",
        "max_steps": 4,
    },
    {
        "name": "Amazon Search",
        "url": "https://www.amazon.com/",
        "task": "Search for 'wireless mouse' in the search bar",
        "category": "medium",
        "max_steps": 3,
    },
    {
        "name": "W3Schools Form",
        "url": "https://www.w3schools.com/html/tryit.asp?filename=tryhtml_form_submit",
        "task": "In the result frame, type 'John' in the First name field and click Submit",
        "category": "medium",
        "max_steps": 4,
    },
    {
        "name": "OpenStreetMap",
        "url": "https://www.openstreetmap.org/",
        "task": "Search for 'New York City' in the search box and click the first result",
        "category": "medium",
        "max_steps": 4,
    },
    {
        "name": "IMDb Search",
        "url": "https://www.imdb.com/",
        "task": "Search for 'Inception' movie and click on the first search result",
        "category": "medium",
        "max_steps": 4,
    },
    {
        "name": "NPM Package Search",
        "url": "https://www.npmjs.com/",
        "task": "Search for 'express' package and click on the first result",
        "category": "medium",
        "max_steps": 4,
    },
    {
        "name": "Weather Check",
        "url": "https://www.weather.gov/",
        "task": "Type 'New York, NY' in the search or location box and search for weather",
        "category": "medium",
        "max_steps": 4,
    },
    {
        "name": "Bing Search + Click",
        "url": "https://www.bing.com/",
        "task": "Search for 'large language models' and click the first result link",
        "category": "medium",
        "max_steps": 4,
    },

    # ======== HARD (10 tasks) — multi-step flows, forms, navigation ========
    {
        "name": "Demo Web Shop Cart",
        "url": "http://demowebshop.tricentis.com/",
        "task": "Navigate to Books category, click on a book, and add it to the shopping cart",
        "category": "hard",
        "max_steps": 5,
    },
    {
        "name": "Booking.com Search",
        "url": "https://www.booking.com/",
        "task": "Search for hotels in 'Paris' with check-in tomorrow and check-out day after tomorrow",
        "category": "hard",
        "max_steps": 5,
    },
    {
        "name": "GitHub Repo Navigation",
        "url": "https://github.com/microsoft/vscode",
        "task": "Go to the Issues tab, click on the first open issue, and scroll to read comments",
        "category": "hard",
        "max_steps": 5,
    },
    {
        "name": "eBay Product Search",
        "url": "https://www.ebay.com/",
        "task": "Search for 'mechanical keyboard', select 'Buy It Now' filter, and click the first result",
        "category": "hard",
        "max_steps": 5,
    },
    {
        "name": "Wikipedia Multi-hop",
        "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "task": "Click on the 'Guido van Rossum' link in the article, then click on his birth city link",
        "category": "hard",
        "max_steps": 5,
    },
    {
        "name": "Target Store Browse",
        "url": "https://www.target.com/",
        "task": "Navigate to Electronics category, then select Headphones, and click on the first product",
        "category": "hard",
        "max_steps": 5,
    },
    {
        "name": "ArXiv Paper Search",
        "url": "https://arxiv.org/",
        "task": "Search for 'web navigation agent' papers, then click on the first result to view the abstract",
        "category": "hard",
        "max_steps": 5,
    },
    {
        "name": "Google Maps Directions",
        "url": "https://www.google.com/maps",
        "task": "Search for 'Times Square New York' and click on the directions button",
        "category": "hard",
        "max_steps": 5,
    },
    {
        "name": "Best Buy Product",
        "url": "https://www.bestbuy.com/",
        "task": "Search for 'laptop', apply the 'Price Low to High' sort, and click the first product",
        "category": "hard",
        "max_steps": 5,
    },
    {
        "name": "Coursera Course Search",
        "url": "https://www.coursera.org/",
        "task": "Search for 'machine learning', click on the first course result, and scroll to the syllabus",
        "category": "hard",
        "max_steps": 5,
    },
]


# ── Result storage ───────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "live_benchmark"


def save_site_result(run_dir: Path, task_entry: Dict, steps: List[StepResult],
                     elapsed: float, error: str = "") -> Dict:
    """Save per-site result JSON and return summary dict."""
    executed = sum(1 for s in steps if s.executed)
    total = len(steps)
    success_rate = (executed / total * 100) if total else 0.0

    summary = {
        "name": task_entry["name"],
        "url": task_entry["url"],
        "task": task_entry["task"],
        "category": task_entry["category"],
        "max_steps": task_entry["max_steps"],
        "total_steps": total,
        "steps_executed": executed,
        "steps_failed": total - executed,
        "success_rate_pct": round(success_rate, 1),
        "elapsed_seconds": round(elapsed, 1),
        "error": error,
        "steps": [],
    }

    for s in steps:
        summary["steps"].append({
            "step": s.step_idx,
            "sub_task": s.sub_task,
            "action": s.action_type,
            "value": s.action_value,
            "dom_nodes": s.num_dom_nodes,
            "candidates": s.num_candidates,
            "grounded_uid": s.grounded_uid,
            "grounded_element": s.grounded_element_summary,
            "confidence": s.confidence,
            "reasoning": s.reasoning,
            "executed": s.executed,
            "selector": s.executed_selector,
        })

    # Sanitize name for filename
    safe_name = task_entry["name"].replace(" ", "_").replace("/", "_")
    out_path = run_dir / f"{safe_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def save_run_summary(run_dir: Path, all_summaries: List[Dict], total_elapsed: float):
    """Save aggregate benchmark summary."""
    n = len(all_summaries)
    by_cat = {}
    for s in all_summaries:
        cat = s["category"]
        by_cat.setdefault(cat, []).append(s)

    cat_stats = {}
    for cat, items in sorted(by_cat.items()):
        total_steps = sum(i["total_steps"] for i in items)
        exec_steps = sum(i["steps_executed"] for i in items)
        failed_sites = sum(1 for i in items if i.get("error"))
        cat_stats[cat] = {
            "sites": len(items),
            "total_steps": total_steps,
            "steps_executed": exec_steps,
            "step_success_rate_pct": round(exec_steps / max(total_steps, 1) * 100, 1),
            "failed_sites": failed_sites,
        }

    total_steps = sum(s["total_steps"] for s in all_summaries)
    total_exec = sum(s["steps_executed"] for s in all_summaries)

    use_local = os.getenv("GROUNDER_USE_LOCAL", "").strip() in {"1", "true", "yes"}
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "grounder": "local:Qwen2.5-0.5B-Prune4Web" if use_local else os.getenv("PRUNE4WEB_GROUNDER_MODEL", "gpt-4o"),
            "planner": os.getenv("PRUNE4WEB_PLANNER_MODEL", "gpt-4o"),
            "filter": os.getenv("PRUNE4WEB_FILTER_MODEL", "gpt-4o"),
        },
        "overall": {
            "total_sites": n,
            "total_steps": total_steps,
            "steps_executed": total_exec,
            "step_success_rate_pct": round(total_exec / max(total_steps, 1) * 100, 1),
            "total_elapsed_seconds": round(total_elapsed, 1),
        },
        "by_category": cat_stats,
        "sites": [{k: v for k, v in s.items() if k != "steps"} for s in all_summaries],
    }

    out_path = run_dir / "benchmark_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nBenchmark summary saved -> {out_path}")
    return report


# ── Pretty printing ──────────────────────────────────────────────────────
def print_benchmark_summary(report: Dict):
    print("\n" + "=" * 72)
    print("PRUNE4WEB LIVE BENCHMARK RESULTS")
    print("=" * 72)

    overall = report["overall"]
    print(f"  Total sites      : {overall['total_sites']}")
    print(f"  Total steps      : {overall['total_steps']}")
    print(f"  Steps executed   : {overall['steps_executed']}")
    print(f"  Step success rate: {overall['step_success_rate_pct']}%")
    print(f"  Total time       : {overall['total_elapsed_seconds']:.0f}s")

    print(f"\n  {'Category':<10} {'Sites':>6} {'Steps':>6} {'Executed':>9} {'Rate':>8}")
    print(f"  {'-'*10} {'-'*6} {'-'*6} {'-'*9} {'-'*8}")
    for cat, stats in report["by_category"].items():
        print(f"  {cat:<10} {stats['sites']:>6} {stats['total_steps']:>6} "
              f"{stats['steps_executed']:>9} {stats['step_success_rate_pct']:>7}%")

    print(f"\n  Per-site results:")
    for s in report["sites"]:
        mark = "OK" if s["steps_executed"] > 0 and not s.get("error") else "X "
        print(f"    [{mark}] {s['category']:<6} {s['name']:<30} "
              f"exec={s['steps_executed']}/{s['total_steps']} "
              f"rate={s['success_rate_pct']}% time={s['elapsed_seconds']}s")

    print("=" * 72)


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


    p = argparse.ArgumentParser(description="Prune4Web live benchmark on 30 websites")
    p.add_argument("--categories", nargs="+", default=["easy", "medium", "hard"],
                   choices=["easy", "medium", "hard"],
                   help="Which difficulty categories to run (default: all)")
    p.add_argument("--max-steps", type=int, default=0,
                   help="Override max steps per task (0 = use per-task default)")
    p.add_argument("--headless", action="store_true",
                   help="Run browser in headless mode")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--use-local-grounder", dest="use_local_grounder", action="store_true",
                      help="Use the locally fine-tuned Qwen2.5-0.5B grounder (default)")
    group.add_argument("--no-use-local-grounder", dest="use_local_grounder", action="store_false",
                      help="Do not use the locally fine-tuned Qwen2.5-0.5B grounder")
    p.set_defaults(use_local_grounder=True)
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for shuffling tasks")
    p.add_argument("--sites", nargs="+", default=[],
                   help="Run only specific sites by name (substring match)")
    args = p.parse_args()

    # Optionally enable local grounder
    if args.use_local_grounder:
        os.environ["GROUNDER_USE_LOCAL"] = "1"
    else:
        os.environ.pop("GROUNDER_USE_LOCAL", None)

    # Check OpenAI key (needed for planner + filter even with local grounder)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env (needed for planner + filter).")
        sys.exit(1)

    # Initialize OpenAI client
    from Prune4Web.run_prune4web import OpenAI as _OAI
    import Prune4Web.run_prune4web as _rpw
    _rpw._CLIENT = _OAI(api_key=api_key)

    # Filter and shuffle tasks
    tasks = [t for t in BENCHMARK_TASKS if t["category"] in args.categories]
    if args.sites:
        tasks = [t for t in tasks
                 if any(s.lower() in t["name"].lower() for s in args.sites)]

    rng = random.Random(args.seed)
    rng.shuffle(tasks)

    if not tasks:
        print("No tasks match the selected filters.")
        sys.exit(0)

    # Create output directory
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    use_local = os.getenv("GROUNDER_USE_LOCAL", "").strip() in {"1", "true", "yes"}

    print("=" * 72)
    print("PRUNE4WEB LIVE BENCHMARK")
    print("=" * 72)
    print(f"  Categories   : {', '.join(args.categories)}")
    print(f"  Total tasks  : {len(tasks)}")
    print(f"  Headless     : {args.headless}")
    print(f"  Grounder     : {'LOCAL Qwen2.5-0.5B' if use_local else 'OpenAI API'}")
    print(f"  Output dir   : {run_dir}")
    print(f"  Shuffled     : seed={args.seed}")
    print("=" * 72)

    # Show task order
    for i, t in enumerate(tasks, 1):
        print(f"  {i:>2}. [{t['category']:<6}] {t['name']}")
    print()

    # Run each task
    all_summaries = []
    t_total_start = time.time()

    for idx, task_entry in enumerate(tasks, 1):
        name = task_entry["name"]
        url = task_entry["url"]
        task_text = task_entry["task"]
        max_steps = args.max_steps if args.max_steps > 0 else task_entry["max_steps"]
        cat = task_entry["category"]

        print(f"\n{'='*72}")
        print(f"  [{idx}/{len(tasks)}] [{cat.upper()}] {name}")
        print(f"  URL:  {url}")
        print(f"  Task: {task_text}")
        print(f"  Max steps: {max_steps}")
        print(f"{'='*72}")

        t_start = time.time()
        steps = []
        error = ""

        try:
            steps = asyncio.run(
                run_prune4web_live(
                    url=url,
                    task=task_text,
                    max_steps=max_steps,
                    headless=args.headless,
                )
            )
            print_step_results(steps)
        except KeyboardInterrupt:
            print(f"\n  Interrupted at {name}. Saving results collected so far...")
            error = "interrupted"
        except Exception as e:
            print(f"\n  ERROR on {name}: {e}")
            error = str(e)[:200]

        elapsed = time.time() - t_start
        summary = save_site_result(run_dir, task_entry, steps, elapsed, error)
        all_summaries.append(summary)

        executed = sum(1 for s in steps if s.executed)
        print(f"\n  [{name}] {executed}/{len(steps)} steps executed in {elapsed:.1f}s")

        if error == "interrupted":
            break

    total_elapsed = time.time() - t_total_start

    # Save and print aggregate report
    report = save_run_summary(run_dir, all_summaries, total_elapsed)
    print_benchmark_summary(report)

    # Print LLM usage
    USAGE_TRACKER.print_summary()

    print(f"\nAll results saved to: {run_dir}")


if __name__ == "__main__":
    main()
