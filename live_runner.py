"""
Live Runner — Run SeeAct pipeline on live websites or Mind2Web dataset.

All configuration is read from ``.env`` (via ``config.py``) — no CLI flags needed.

Key .env variables::

    RUNNER_MODE=live              # "live" or "dataset"
    USE_GPT=true                  # true = GPT-4o API, false = local Qwen
    LIVE_SITES_PATH=src/live_sites.json
    BROWSER_HEADLESS=false

    # Dataset mode only
    ANNOTATION_ID=                # leave empty for batch mode
    DATASET_FILE=train-00002-of-00027-....parquet
    FORCE_REPROCESS=false

Usage:
    python live_runner.py

Requirements (live mode only):
    pip install playwright
    playwright install chromium
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Optional

# ── path setup ────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from AutoWeb.src.config import (
    get_model_path, get_openai_model,
    get_runner_mode, get_use_gpt, get_live_sites_path,
    get_browser_headless, get_annotation_id, get_dataset_file,
    get_force_reprocess, print_config,
)
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


def run_live_mode():
    """
    Interactive live-website loop.  All settings read from .env.

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

    use_gpt = get_use_gpt()
    headless = get_browser_headless()
    sites_path = get_live_sites_path()

    sites = _load_sites(sites_path)

    # ── Initialise pipeline (model loads once) ────────────────────
    model_folder = get_model_path()
    backend_label = (
        f"{get_openai_model()} (OpenAI API)" if use_gpt else "Qwen2-VL-2B (local)"
    )
    logger.info("=" * 80)
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

def run_dataset_mode():
    """Delegate to the existing Mind2Web-based pipeline.  All settings from .env."""
    from AutoWeb.src.seeact_pipeline import run_single_prediction_example

    model_folder = get_model_path()
    annotation_id = get_annotation_id()
    dataset_file = get_dataset_file()
    use_gpt = get_use_gpt()
    force_reprocess = get_force_reprocess()

    if not dataset_file and not annotation_id:
        raise ValueError(
            "Dataset mode requires DATASET_FILE and/or ANNOTATION_ID in .env"
        )

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
    """Entry point — reads RUNNER_MODE from .env and dispatches."""
    print_config()
    mode = get_runner_mode()
    logger.info(f"Runner mode: {mode}")

    if mode == "live":
        run_live_mode()
    else:
        run_dataset_mode()


if __name__ == "__main__":
    main()
