"""
Live Runner — Run SeeAct pipeline on live websites with PolicyHub + HITL.

Supports two modes:
    1. **Live mode**  — Opens a real browser, iterates over target websites,
       asks the user for an instruction in the terminal, and runs the
       SeeAct pipeline (Action Gen → HITL Gate → Grounding).
    2. **Dataset mode** — Delegates to the existing Mind2Web-based
       ``run_single_prediction_example()`` for offline evaluation.

Usage:
    # Live mode  (default — opens browser)
    python live_runner.py --mode live

    # Live mode with custom sites file
    python live_runner.py --mode live --sites src/live_sites.json

    # Dataset mode (existing Mind2Web workflow)
    python live_runner.py --mode dataset --annotation_id "abc123" --dataset_file "test_task_0.parquet"

    # Dataset mode — batch (all annotation_ids in a parquet)
    python live_runner.py --mode dataset --dataset_file "test_task_0.parquet"

Requirements (live mode only):
    pip install playwright
    playwright install chromium
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# ── path setup ────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from AutoWeb.src.config import get_model_path, get_device, get_openai_model
from AutoWeb.src.logger import logger


# ======================================================================
# Live-mode runner
# ======================================================================

def _load_sites(sites_path: str) -> List[Dict]:
    """Load the target-sites list from a JSON file."""
    path = Path(sites_path)
    if not path.exists():
        raise FileNotFoundError(f"Sites config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sites = data.get("sites", [])
    logger.info(f"Loaded {len(sites)} sites from {path}")
    return sites


def run_live_mode(
    sites_path: str,
    use_gpt: bool = False,
    headless: bool = False,
):
    """
    Interactive live-website loop.

    For each site:
        1. Navigate browser to the URL.
        2. Ask the user for an instruction in the terminal.
        3. Capture screenshot + DOM.
        4. Run SeeAct pipeline  (Action Gen → HITL → Grounding).
        5. Display results and ask if another instruction is desired.
    """
    from AutoWeb.src.browser_driver import BrowserDriver
    from AutoWeb.src.seeact_pipeline import SeeActPipeline
    from AutoWeb.src.live_result_store import LiveResultStore

    sites = _load_sites(sites_path)

    # ── Initialise pipeline (model loads once) ────────────────────
    model_folder = get_model_path()
    logger.info("=" * 80)
    backend_label = (
        f"{get_openai_model()} (OpenAI API)" if use_gpt else "Qwen2-VL-2B (local)"
    )
    logger.info("SeeAct Live Runner")
    logger.info(f"  Model backend : {backend_label}")
    logger.info(f"  Sites loaded  : {len(sites)}")
    logger.info("=" * 80)

    pipeline = SeeActPipeline(
        model_folder=model_folder,
        use_gpt=use_gpt,
    )

    # ── Initialise browser ────────────────────────────────────────
    driver = BrowserDriver(headless=headless)

    # ── Initialise result store (screenshots + JSON per step) ─────
    result_store = LiveResultStore(base_dir="liveSiteResults")

    all_results: List[Dict] = []

    try:
        for idx, site in enumerate(sites, start=1):
            url = site["url"]
            domain = site.get("domain", "")
            name = site.get("name", url)
            default_ins = site.get("default_instruction")

            print(f"\n{'='*70}")
            print(f"  [{idx}/{len(sites)}]  {name}  —  {url}")
            print(f"{'='*70}")

            # Navigate
            driver.navigate(url)

            # ── Per-site instruction loop ─────────────────────────
            action_history = ["None"]
            step_num = 0

            while True:
                step_num += 1

                # Ask user for instruction (or use default on first step)
                if default_ins and step_num == 1:
                    instruction = default_ins
                    print(f"\n  Using default instruction: {instruction}")
                    print("  (Press Enter to accept, or type a new one)")
                    override = input("  > ").strip()
                    if override:
                        instruction = override
                else:
                    print(f"\n  Step {step_num} — Enter instruction for {name}")
                    print("  (type 'skip' to move to next site, 'quit' to exit)")
                    instruction = input("  > ").strip()

                if instruction.lower() == "quit":
                    print("\n  Exiting live runner.")
                    driver.close()
                    _print_live_summary(all_results)
                    result_store.save_run_summary(all_results)
                    return all_results

                if instruction.lower() == "skip":
                    print(f"  Skipping {name}.")
                    break

                if not instruction:
                    print("  Empty instruction — skipping step.")
                    continue

                # Capture current page state
                print("  Capturing page state...")
                state = driver.capture_state()

                # Run pipeline prediction
                print("  Running SeeAct pipeline...")
                result = pipeline.predict_live_step(
                    screenshot=state["screenshot"],
                    cleaned_html=state["cleaned_html"],
                    instruction=instruction,
                    website=domain,
                    action_history=action_history,
                )

                # Display result
                _print_step_result(result, name, step_num)

                # Accumulate
                result["site"] = name
                result["url"] = url
                result["instruction"] = instruction
                result["step_num"] = step_num
                all_results.append(result)

                # Persist screenshot + result JSON
                result_store.save_step(
                    site_name=name,
                    step_num=step_num,
                    screenshot=state["screenshot"],
                    result=result,
                )

                # Update history for next step
                if result.get("output_plan"):
                    action_history.append({"action_plan": result["output_plan"]})

                # Ask to continue with another instruction on same site
                print("\n  Another instruction on this site? (Enter = yes, 'next' = next site, 'quit' = exit)")
                cont = input("  > ").strip().lower()
                if cont == "quit":
                    print("\n  Exiting live runner.")
                    driver.close()
                    _print_live_summary(all_results)
                    result_store.save_run_summary(all_results)
                    return all_results
                if cont == "next":
                    break
                # else: continue loop for another instruction

    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")
    finally:
        driver.close()

    _print_live_summary(all_results)
    result_store.save_run_summary(all_results)
    return all_results


# ======================================================================
# Pretty-print helpers
# ======================================================================

def _print_step_result(result: Dict, site_name: str, step_num: int):
    """Display a single live-step result in the terminal."""
    print(f"\n  {'─'*60}")
    print(f"  Result  [{site_name} step {step_num}]")
    print(f"  {'─'*60}")

    if not result.get("success"):
        print(f"  ✗ FAILED: {result.get('error', 'unknown')}")
        return

    plan = result.get("output_plan", "(empty)")
    conf = result.get("composite_confidence", 0)
    llm_conf = result.get("llm_confidence", 0)
    hitl = result.get("hitl_triggered", False)
    reason = result.get("hitl_reason", "")

    print(f"  Action Plan    : {plan}")
    print(f"  LLM Confidence : {llm_conf:.1f}")
    print(f"  Composite Conf : {conf:.1f}")
    print(f"  HITL Triggered : {'⚡ YES' if hitl else '✓ NO'}")
    if hitl and reason:
        print(f"  HITL Reason    : {reason}")

    gr = result.get("grounding_result") or {}
    if gr.get("success"):
        sel = gr.get("selected_element", {})
        tag = sel.get("tag", "?")
        text = (sel.get("text") or "")[:60]
        print(f"  Grounded To    : <{tag}> '{text}'")
    elif gr.get("error") == "hitl_triggered":
        print(f"  Grounding      : SKIPPED (HITL triggered)")
    else:
        print(f"  Grounding      : ✗ Failed — {gr.get('error', 'unknown')}")

    cost = result.get("cost", {})
    if cost:
        print(f"  Step Cost      : ${cost.get('cost_usd', 0):.6f}")

    print(f"  Latency        : {result.get('latency', 0):.2f}s")


def _print_live_summary(results: List[Dict]):
    """Print an aggregate summary of all live-mode steps."""
    if not results:
        print("\n  No results to summarise.")
        return

    total = len(results)
    successful = sum(1 for r in results if r.get("success"))
    hitl_count = sum(1 for r in results if r.get("hitl_triggered"))
    avg_conf = (
        sum(r.get("composite_confidence", 0) for r in results) / total
        if total else 0
    )

    print(f"\n{'='*70}")
    print("  LIVE RUN SUMMARY")
    print(f"{'='*70}")
    print(f"  Total steps       : {total}")
    print(f"  Successful        : {successful}")
    print(f"  HITL triggers     : {hitl_count}  ({hitl_count/total:.0%})")
    print(f"  Avg confidence    : {avg_conf:.1f}")

    # Per-site breakdown
    sites_seen = {}
    for r in results:
        s = r.get("site", "?")
        sites_seen.setdefault(s, []).append(r)

    print(f"\n  Per-site breakdown:")
    for s, steps in sites_seen.items():
        n_hitl = sum(1 for r in steps if r.get("hitl_triggered"))
        avg_c = sum(r.get("composite_confidence", 0) for r in steps) / len(steps)
        print(f"    {s:20s}  steps={len(steps)}  hitl={n_hitl}  avg_conf={avg_c:.1f}")

    print(f"{'='*70}\n")


# ======================================================================
# Dataset-mode (existing workflow)
# ======================================================================

def run_dataset_mode(
    annotation_id: Optional[str],
    dataset_file: Optional[str],
    use_gpt: bool = False,
    force_reprocess: bool = False,
):
    """Delegate to the existing Mind2Web-based pipeline."""
    from AutoWeb.src.seeact_pipeline import run_single_prediction_example

    model_folder = get_model_path()
    return run_single_prediction_example(
        model_folder=model_folder,
        annotation_id=annotation_id,
        dataset_file_name=dataset_file,
        use_gpt=use_gpt,
        force_reprocess=force_reprocess,
    )


# ======================================================================
# CLI entry point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SeeAct Pipeline Runner — Live websites or Mind2Web dataset"
    )
    parser.add_argument(
        "--mode",
        choices=["live", "dataset"],
        default="live",
        help="Run mode: 'live' (browser + interactive) or 'dataset' (Mind2Web offline).",
    )

    # Live-mode options
    parser.add_argument(
        "--sites",
        default=str(SRC_DIR / "live_sites.json"),
        help="Path to live_sites.json config file.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (no visible window).",
    )

    # Dataset-mode options
    parser.add_argument("--annotation_id", default=None, help="Specific annotation ID (dataset mode).")
    parser.add_argument("--dataset_file", default=None, help="Parquet file name (dataset mode).")
    parser.add_argument("--force_reprocess", action="store_true", help="Re-process already-checkpointed tasks.")

    # Shared options
    parser.add_argument("--gpt", action="store_true", help="Use GPT-4o (OpenAI API) instead of local Qwen model.")

    args = parser.parse_args()

    if args.mode == "live":
        run_live_mode(
            sites_path=args.sites,
            use_gpt=args.gpt,
            headless=args.headless,
        )
    else:
        if not args.dataset_file and not args.annotation_id:
            parser.error("Dataset mode requires --dataset_file and/or --annotation_id.")
        run_dataset_mode(
            annotation_id=args.annotation_id,
            dataset_file=args.dataset_file,
            use_gpt=args.gpt,
            force_reprocess=args.force_reprocess,
        )


if __name__ == "__main__":
    main()
