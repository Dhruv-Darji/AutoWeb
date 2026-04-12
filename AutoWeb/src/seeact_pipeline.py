"""
Final plan available at AutoWeb/Docs/SeeAct_understanding.md
"""

import os
import sys
from pathlib import Path
import time
from typing import Dict, Optional

from AutoWeb.src.action_generation import SeeActActionGenerator
from AutoWeb.src.action_grounding import SeeActActionGrounding
from AutoWeb.src.seeact_evaluation import SeeActEvaluator


# Add src to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from AutoWeb.src.Input_Prepration import SeeActInputPreparator
from config import get_model_path, get_device, get_model_dtype, get_use_8bit, get_openai_model, get_policy_hub_path, get_hitl_threshold, get_low_confidence_floor, get_skip_grounding, get_runner_mode
from AutoWeb.src.model_interface import VLModel
from AutoWeb.src.gpt_model import GPTVisionModel
from AutoWeb.src.logger import logger
from AutoWeb.src.utils.processed_store import ProcessedStore
from AutoWeb.src.policyHub import PolicyHub, HITLConfidenceGate


# New imports:
from mind2Web_Loader import Mind2WebDataset

class SeeActPipeline:
    """
    Complete SeeAct-style single-Task UI action prediction pipeline.
    
    This class integrates:
    - Dataset Loading
    - Input prepration (Prompt Template + Task , Screenshot Image (i), History of Previous Actions for task)        
    - Action Generation (take input from previous block, result a planning for task)
    - Action Grounding (3 methods: Element Attributes, Textual Choices, Image Annotation)
    - Action decoding (as per user method selection retrieve the output)
    - Evaluation
    
    Input: screenshot (PIL.Image) + instruction (str) + Cleaned HTML
    Output: Action to be perform with target
    """
    
    def __init__(self,
                 model_folder: str,
                 device: Optional[str] = None,
                 use_gpt: bool = False):
        logger.info("[SeeActPipeline] Initializing components...")
        self.use_gpt = use_gpt
        self.MAX_BATCH = 15
        
        # 1. Mind2Web Dataset Loader (for retrieving task data based on annotation ID)
        self.mind2web_loader = Mind2WebDataset(root_dir="D:\\Environments\\Datasets\\multimodal-mind2web")
        logger.info(f"  ✓ Mind2Web dataset loader ready(root: D:\\Environments\\Datasets\\multimodal-mind2web)")
        
        # 2. Prompt Engine (strict JSON enforcement)
        self.input_preparator = SeeActInputPreparator()
        logger.info("  ✓ Input preparator ready")
        
        # 3. Model Interface — GPT-4o (API) or local Qwen2-VL-2B
        if self.use_gpt:
            logger.info("  ⏳ Initializing GPT-4o model (OpenAI API)...")
            self.model = GPTVisionModel()
            logger.info("  ✓ GPT-4o model ready for inference")
        else:
            logger.info(f"  ⏳ Loading local Qwen model from {model_folder}...")
            # decide device: prefer explicit argument, else fall back to config.get_device()
            device_to_use = device or get_device()
            try:
                self.model = VLModel(
                    model_folder=model_folder,
                    device=device_to_use
                )
                logger.info("  ✓ Qwen model loaded and ready for inference")
            except Exception as e:
                logger.exception(f"[SeeActPipeline] Error loading model: {e}")
                try:
                    import torch
                    logger.debug(f"[SeeActPipeline] torch.cuda.is_available(): {torch.cuda.is_available()}")
                    if torch.cuda.is_available():
                        try:
                            logger.debug(f"[SeeActPipeline] cuda memory allocated: {torch.cuda.memory_allocated(0)}")
                            logger.debug(f"[SeeActPipeline] cuda memory reserved: {torch.cuda.memory_reserved(0)}")
                        except Exception:
                            pass
                except Exception:
                    pass
                logger.warning("[SeeActPipeline] Falling back to CPU model load (this may be slower).")
                self.model = VLModel(model_folder=model_folder, device="cpu")
                logger.info("  ✓ Model loaded on CPU (fallback)")
        

        # 4. Action Generator (SeeAct Action Generation Module)
        logger.info("  ⏳ Initializing action generator...")
        self.actionGenration = SeeActActionGenerator(model=self.model)
        logger.info("  ✓ Action generator ready")

        # 5. Action Grounding (SeeAct Action Grounding Module)
        self.skip_grounding = get_skip_grounding()
        if self.skip_grounding:
            self.action_grounding = None
            logger.info("  ⏩ Action grounding SKIPPED (SKIP_GROUNDING=true)")
        else:
            self.action_grounding = SeeActActionGrounding(model=self.model)
        
        # 6. Evaluator (SeeAct offline metrics)
        self.evaluator = SeeActEvaluator(output_dir="eval_results")
        logger.info("  ✓ Evaluator ready")

        # 7. PolicyHub + HITL Confidence Gate (risk-based)
        self.policy_hub = PolicyHub(json_path=get_policy_hub_path())
        self.hitl_gate = HITLConfidenceGate(
            low_confidence_floor=get_low_confidence_floor(),
            threshold=get_hitl_threshold(),
        )
        logger.info(f"  ✓ PolicyHub loaded ({self.policy_hub.get_stats()})")
        logger.info(f"  ✓ HITL gate ready (risk-based, low_confidence_floor={get_low_confidence_floor()})")
        
        logger.info("[SeeActPipeline] ✓ Pipeline ready!")
    
    def predict_single_task(
            self, 
            annotation_id: str, 
            dataset_file_name: str,
            seeAct_method: str = "2" #1. Element Attributes 2. Textual Choice 3. Image Annotation
        ) -> Dict:
        # Ensure evaluator writes into a subdirectory named for the dataset file.
        if dataset_file_name:
            base = "eval_results"
            sub = os.path.basename(dataset_file_name)
            out_dir = os.path.join(base, sub)
            os.makedirs(out_dir, exist_ok=True)
            self.evaluator.output_dir = out_dir
            logger.info(f"Evaluator output redirected to {out_dir}")
        """
        Run the full SeeAct-style prediction for a single task (multiple steps).
        Available methods for action grounding:
        1. Element Attributes
        2. Textual Choice
        3. Image Annotation
        """

        logger.info(f"\n[SeeActPipeline] Predicting action for: '{annotation_id}'")
        
        # Step 1: Load single task data (multiple steps (with details like screenshot, cleaned HTML) + instruction)
        logger.info("="*20 + " [1/6] Loading task data... " + "="*20)
        single_task = self.mind2web_loader.get_task(
            annotation_id=annotation_id,
            file_name=dataset_file_name
        )
        
        logger.info(f"Instruction: {single_task[0]['instruction']}")

        # PolicyHub: resolve active policy for this task's website
        task_website = single_task[0].get("website", None)
        policy_text = self.policy_hub.get_active_policy(domain=task_website)
        policy_risk_keywords = self.policy_hub.get_policy_risk_keywords()
        logger.info(f"  PolicyHub: domain='{task_website}', policy length={len(policy_text)} chars, risk_keywords={len(policy_risk_keywords)}")
        
        logger.info("="*20 + f" For each action in Task {annotation_id}... " + "="*20)

        task_latency_start = time.time()

        # Initialize history of previous actions for the task (if needed for input preparation)
        action_history = ["None"]

        for action in single_task:
            logger.info(f"\n--- Processing action: '{action['action_uid']}' ---")
            action_latency_start = time.time()

            # Reset per-step cost tracker (GPT mode only)
            if self.use_gpt and hasattr(self.model, 'reset_step_cost'):
                self.model.reset_step_cost()

            # Step 2: Prepare inputs for single task (stepImage + past history + instruction)
            logger.info("="*20 + " [2/6] Input Preparation... " + "="*20)
            
            action_generation_input = self.input_preparator.prepare_input(
                action=action,
                history=action_history,
                policy_text=policy_text,
            )
        
            # Step 3: Action Generation (take input from previous block, result a planning for task)
            logger.info("="*20 + " [3/6] Running Action Generation... " + "="*20)
            action_generation_plan = self.actionGenration.generate_plans(
                input_data=action_generation_input
            )

            output_plan = action_generation_plan.get("output_text", "")
            llm_confidence = action_generation_plan.get("confidence", 0.0)
            policy_risk_flag = action_generation_plan.get("policy_risk", False)
            hitl_reason_from_llm = action_generation_plan.get("hitl_reason", "")
            err = action_generation_plan.get("error")

            logger.info(f"Generated Action Plan: {output_plan}")
            logger.info(f"  LLM confidence: {llm_confidence}, policy_risk: {policy_risk_flag}, hitl_reason: {hitl_reason_from_llm}")

            if err:
                logger.error(f"✗ Action generation failed with error: {err}")
                # continue to next action or decide how to handle this case (e.g., skip grounding/decoding for this step)
                continue

            # store the input and action plan in history for potential use in future steps
            # in dataset mode, we append the **ground truth** plan from the
            # annotated data instead of the model's prediction.  this prevents
            # a single mistaken generation from polluting all subsequent steps
            # and ensures we test the downstream grounding/architecture.
            if get_runner_mode() == "dataset":
                gt_plan = action.get("target_action_reprs") or output_plan
                action_history.append({"action_plan": gt_plan})
            else:
                action_history.append({"action_plan": output_plan})

            # Step 3.5: HITL Confidence Gate (runs BEFORE grounding to save cost)
            hitl_result = self.hitl_gate.compute_composite_confidence(
                llm_confidence=llm_confidence,
                action_text=output_plan,
                policy_risk_keywords=policy_risk_keywords,
                policy_risk_flag=policy_risk_flag,
                hitl_reason=hitl_reason_from_llm,
            )
            composite_confidence = hitl_result["final_confidence"]
            hitl_triggered = hitl_result["hitl_triggered"]
            hitl_reason = hitl_reason_from_llm or "; ".join(hitl_result.get("trigger_reasons", []))

            if hitl_triggered:
                logger.warning(
                    f"  \u26a1 HITL TRIGGERED \u2014 skipping grounding  "
                    f"reasons={hitl_result.get('trigger_reasons', [])}  "
                    f"llm_conf={llm_confidence}  risk_flag={policy_risk_flag}  "
                    f"keywords={hitl_result.get('matched_keywords', [])}"
                )
                # Skip expensive grounding call; record a stub result
                grounding_result = {"success": False, "error": "hitl_triggered", "selected_element": None}

            # Skip grounding if plan is empty — record as failed step directly
            elif not output_plan.strip():
                logger.warning("    ⚠ Empty action plan — skipping grounding, recording as failed step.")
                grounding_result = {"success": False, "error": "empty plan", "selected_element": None}            
            elif self.skip_grounding:
                logger.info(f"  ✓ No risk detected  llm_conf={llm_confidence}, risk_flag={policy_risk_flag}")
                logger.info("  ⏩ Grounding SKIPPED (SKIP_GROUNDING=true)")
                grounding_result = {"success": False, "error": "grounding_skipped", "selected_element": None}            
            else:
                logger.info(f"  \u2713 No risk detected  llm_conf={llm_confidence}, risk_flag={policy_risk_flag}")
                # Step 4: Grounding method selection and processing
                logger.info("="*20 + " [4/6] Running Action Grounding ... " + "="*20)
                grounding_result = self.action_grounding.process(
                    annotation_id=annotation_id,
                    textual_plan=output_plan,
                    step_info=action)

            # Step 5: Parse the Grounding output 
            logger.info("="*20 + " [5/6] Parsing output... " + "="*20)
            grounding_result = grounding_result or {}
            if grounding_result.get("success"):
                sel = grounding_result.get("selected_element", {})
                logger.info(
                    f"        ✓ Grounded to: <{sel.get('tag', '?')}> "
                    f"text='{(sel.get('text') or '')[:60]}' "
                    f"bid={((sel.get('attributes') or {}).get('backend_node_id', '?'))}"
                )
            else:
                logger.error(f"        ✗ Grounding failed: {grounding_result.get('error', 'unknown')}")

            # Step 6: Evaluate against ground truth
            logger.info("="*20 + " [6/6] Evaluating prediction... " + "="*20)
            action_latency_end = time.time()
            action_latency = action_latency_end - action_latency_start

            self.evaluator.record_step(
                annotation_id=annotation_id,
                action_uid=action.get("action_uid", ""),
                ground_truth=action,
                predicted_plan=output_plan,
                grounding_result=grounding_result,
                latency=action_latency,
                llm_confidence=llm_confidence,
                composite_confidence=composite_confidence,
                policy_risk_flag=policy_risk_flag,
                hitl_triggered=hitl_triggered,
                hitl_reason=hitl_reason,
            )

            # Print per-step GPT cost
            if self.use_gpt and hasattr(self.model, 'get_step_cost'):
                sc = self.model.get_step_cost()
                logger.info(
                    f"        💰 Step cost: ${sc['cost_usd']:.6f} "
                    f"({sc['prompt_tokens']} in + {sc['completion_tokens']} out = "
                    f"{sc['total_tokens']} tokens, {sc['calls']} API calls)"
                )

        # Compute and save evaluation metrics for this task
        self.evaluator.compute_metrics()
        self.evaluator.print_summary()
        saved = self.evaluator.save_results(tag=annotation_id)

        task_latency_end = time.time()
        total_task_latency = task_latency_end - task_latency_start

        # Print task-level GPT cost summary
        task_cost_data = {}
        if self.use_gpt and hasattr(self.model, 'get_total_cost'):
            tc = self.model.get_total_cost()
            task_cost_data = tc
            logger.info("\n" + "=" * 60)
            logger.info(f"💰 TASK GPT COST SUMMARY  (model: {tc['model']})")
            logger.info(f"   Total API calls:      {tc['calls']}")
            logger.info(f"   Prompt tokens:        {tc['prompt_tokens']:,}")
            logger.info(f"   Completion tokens:    {tc['completion_tokens']:,}")
            logger.info(f"   Total tokens:         {tc['total_tokens']:,}")
            logger.info(f"   Total cost:           ${tc['cost_usd']:.6f}")
            logger.info("=" * 60)

        task_eval = self.evaluator.task_results.get(annotation_id)

        return {
            "success": True,
            "error": None,
            "latency": total_task_latency,
            "evaluation": {
                "aggregate": self.evaluator.aggregate.to_dict() if self.evaluator.aggregate else {},
                "task": task_eval.to_dict() if task_eval else {},
                "saved_files": saved,
            },
            "cost": task_cost_data,
        }  

    # ------------------------------------------------------------------
    # Live-mode: predict a single step from a live browser capture
    # ------------------------------------------------------------------

    def predict_live_step(
        self,
        screenshot,
        cleaned_html: str,
        instruction: str,
        website: str = "",
        action_history: list = None,
    ) -> Dict:
        """
        Run one prediction cycle on a live page capture (no ground truth).

        Flow:  Input Prep → Action Gen → HITL Gate → (if passed) Grounding

        Args:
            screenshot:    PIL.Image of the current viewport.
            cleaned_html:  Body HTML string (used for grounding element extraction).
            instruction:   User-specified task instruction.
            website:       Website domain (used for PolicyHub policy lookup).
            action_history: List of previous action plan strings (or None).

        Returns:
            Dict with prediction results including HITL gate outcome.
        """
        import uuid
        step_start = time.time()

        # Build a synthetic action dict matching Mind2Web format
        step_id = str(uuid.uuid4())[:8]
        action = {
            "action_uid": f"live_{step_id}",
            "instruction": instruction,
            "screenshot": screenshot,
            "cleaned_html": cleaned_html,
            "website": website,
        }

        # Reset per-step cost tracker (GPT mode only)
        if self.use_gpt and hasattr(self.model, 'reset_step_cost'):
            self.model.reset_step_cost()

        # PolicyHub: resolve active policy for this website
        policy_text = self.policy_hub.get_active_policy(domain=website)
        policy_risk_keywords = self.policy_hub.get_policy_risk_keywords()
        logger.info(f"  PolicyHub: domain='{website}', policy={len(policy_text)} chars")

        # Step 1: Input Preparation
        logger.info("=" * 20 + " [1/4] Input Preparation... " + "=" * 20)
        history = action_history if action_history else ["None"]
        action_generation_input = self.input_preparator.prepare_input(
            action=action,
            history=history,
            policy_text=policy_text,
        )

        # Step 2: Action Generation
        logger.info("=" * 20 + " [2/4] Running Action Generation... " + "=" * 20)
        action_generation_plan = self.actionGenration.generate_plans(
            input_data=action_generation_input
        )

        output_plan = action_generation_plan.get("output_text", "")
        llm_confidence = action_generation_plan.get("confidence", 0.0)
        policy_risk_flag = action_generation_plan.get("policy_risk", False)
        hitl_reason_from_llm = action_generation_plan.get("hitl_reason", "")
        err = action_generation_plan.get("error")

        logger.info(f"Generated Action Plan: {output_plan}")
        logger.info(f"  LLM confidence: {llm_confidence}, policy_risk: {policy_risk_flag}, hitl_reason: {hitl_reason_from_llm}")

        if err:
            logger.error(f"✗ Action generation failed: {err}")
            return {
                "success": False,
                "error": err,
                "output_plan": "",
                "hitl_triggered": False,
                "composite_confidence": 0.0,
                "grounding_result": None,
                "latency": time.time() - step_start,
            }

        # Step 2.5: HITL Confidence Gate (before grounding to save cost)
        logger.info("=" * 20 + " [2.5/4] HITL Confidence Gate... " + "=" * 20)
        hitl_result = self.hitl_gate.compute_composite_confidence(
            llm_confidence=llm_confidence,
            action_text=output_plan,
            policy_risk_keywords=policy_risk_keywords,
            policy_risk_flag=policy_risk_flag,
            hitl_reason=hitl_reason_from_llm,
        )
        composite_confidence = hitl_result["final_confidence"]
        hitl_triggered = hitl_result["hitl_triggered"]
        hitl_reason = hitl_reason_from_llm or "; ".join(hitl_result.get("trigger_reasons", []))
        

        if hitl_triggered:
            logger.warning(
                f"  ⚡ HITL TRIGGERED — skipping grounding  "
                f"reasons={hitl_result.get('trigger_reasons', [])}  "
                f"llm_conf={llm_confidence}  risk_flag={policy_risk_flag}  "
                f"keywords={hitl_result.get('matched_keywords', [])}"
            )
            grounding_result = {"success": False, "error": "hitl_triggered", "selected_element": None}

        elif not output_plan.strip():
            logger.warning("    ⚠ Empty action plan — skipping grounding.")
            grounding_result = {"success": False, "error": "empty plan", "selected_element": None}
        elif self.skip_grounding:
            logger.info(f"  ✓ No risk detected  llm_conf={llm_confidence}, risk_flag={policy_risk_flag}")
            logger.info("  ⏩ Grounding SKIPPED (SKIP_GROUNDING=true)")
            grounding_result = {"success": False, "error": "grounding_skipped", "selected_element": None}
        else:
            logger.info(f"  \u2713 No risk detected  llm_conf={llm_confidence}, risk_flag={policy_risk_flag}")
            # Step 3: Action Grounding
            logger.info("=" * 20 + " [3/4] Running Action Grounding... " + "=" * 20)
            grounding_result = self.action_grounding.process(
                annotation_id=f"live_{step_id}",
                textual_plan=output_plan,
                step_info=action,
            )

        # Step 4: Parse grounding output
        logger.info("=" * 20 + " [4/4] Parsing output... " + "=" * 20)
        grounding_result = grounding_result or {}
        if grounding_result.get("success"):
            sel = grounding_result.get("selected_element", {})
            logger.info(
                f"        ✓ Grounded to: <{sel.get('tag', '?')}> "
                f"text='{(sel.get('text') or '')[:60]}' "
                f"bid={((sel.get('attributes') or {}).get('backend_node_id', '?'))}"
            )
        else:
            logger.error(f"        ✗ Grounding failed: {grounding_result.get('error', 'unknown')}")

        step_latency = time.time() - step_start

        # Step cost (GPT mode)
        step_cost = {}
        if self.use_gpt and hasattr(self.model, 'get_step_cost'):
            step_cost = self.model.get_step_cost()

        return {
            "success": True,
            "output_plan": output_plan,
            "llm_confidence": llm_confidence,
            "policy_risk_flag": policy_risk_flag,
            "composite_confidence": composite_confidence,
            "hitl_triggered": hitl_triggered,
            "hitl_reason": hitl_reason,
            "hitl_details": hitl_result,
            "grounding_result": grounding_result,
            "latency": step_latency,
            "cost": step_cost,
        }

def run_single_prediction_example(
    model_folder: str,
    annotation_id: Optional[str] = None,
    dataset_file_name: Optional[str] = None,
    device: Optional[str] = None,
    use_gpt: bool = False,
    force_reprocess: bool = False,
):
    """
    Run SeeAct on either a single annotation_id (if provided) or on *all*
    unique annotation_ids found inside `dataset_file_name`.

    Behavior:
      - If `annotation_id` is set (non-empty) -> run single-task pipeline for that id.
      - Else -> run the pipeline for every unique `annotation_id` present in
        the given `dataset_file_name` (batch mode).

    End-of-run: aggregated evaluation is computed & saved for all processed tasks.
    """
    backend_label = (
        f"{get_openai_model()} (OpenAI API)" if use_gpt else "Qwen2-VL-2B (local)"
    )
    logger.info("=" * 80)
    logger.info("SeeAct Single-Step UI Action Predictor")
    logger.info(f"  Model backend: {backend_label}")
    logger.info("=" * 80)

    # Initialize pipeline once so evaluator accumulates across tasks
    pipeline = SeeActPipeline(
        model_folder=model_folder,
        use_gpt=use_gpt,
    )
    # if we know the dataset file name up front, make the evaluator write
    # into a dedicated directory inside eval_results
    if dataset_file_name:
        outdir = os.path.join("eval_results", os.path.basename(dataset_file_name))
        os.makedirs(outdir, exist_ok=True)
        pipeline.evaluator.output_dir = outdir
        logger.info(f"Evaluator output redirected to {outdir}")

    processed_results = []

    # Persistent checkpoint store (skip processed tasks across restarts)
    store = ProcessedStore()
    logger.debug(f"ProcessedStore loaded ({store.count()} entries): {store.path}")

    # Helper to run a single annotation and collect result
    # `use_file=True` means pass `dataset_file_name` to predict_single_task (legacy behavior).
    # In batch mode we use `use_file=False` so predict_single_task will read from
    # the loader's in-memory `df` (avoids re-reading the parquet for each task).
    def _run_one(ann_id: str, use_file: bool = True):
        # If already processed and not forcing re-run, skip immediately
        if not force_reprocess and store.contains(ann_id):
            logger.info(f"Skipping annotation {ann_id}: already processed (checkpoint)")
            skipped_res = {"success": True, "skipped": True, "latency": 0.0}
            processed_results.append((ann_id, skipped_res))
            return skipped_res

        logger.info(f"\n--- Running annotation: {ann_id} ---")
        fn = dataset_file_name if use_file else None
        try:
            res = pipeline.predict_single_task(annotation_id=ann_id, dataset_file_name=fn)
        except Exception as e:
            logger.exception(f"Task {ann_id} failed: {e}")
            failed = {"success": False, "error": str(e)}
            processed_results.append((ann_id, failed))
            return failed

        # Mark processed only on successful completion
        if res.get("success"):
            try:
                store.add(ann_id)
                logger.info(f"Checkpointed annotation: {ann_id}")
            except Exception:
                logger.exception(f"Failed to checkpoint annotation: {ann_id}")

        processed_results.append((ann_id, res))
        return res

    # If a specific annotation_id is provided -> single-run (preserve file arg)
    if annotation_id:
        result = _run_one(annotation_id, use_file=True)

    else:
        # Batch mode: process every unique annotation_id in the specified parquet
        if not dataset_file_name:
            raise ValueError("dataset_file_name must be provided for batch mode (when annotation_id is None)")

        # Find matching parquet(s) in the dataset loader (one-time read)
        matched_paths = [p for p in pipeline.mind2web_loader.parquet_files if p.endswith(dataset_file_name) or os.path.basename(p) == dataset_file_name]
        if not matched_paths:
            # fallback: use the loader's in-memory concatenated DataFrame (already loaded at init)
            df = pipeline.mind2web_loader.df
        else:
            import pandas as _pd
            dfs = [_pd.read_parquet(p) for p in matched_paths]
            df = _pd.concat(dfs, ignore_index=True)

        unique_ids = list(df["annotation_id"].dropna().unique())
        logger.info(f"Batch mode: found {len(unique_ids)} unique annotation_id(s) in '{dataset_file_name}'")

        # Iterate and run pipeline for each annotation id using in-memory df (avoid re-read)
        if pipeline.MAX_BATCH:
            unique_ids = unique_ids[:pipeline.MAX_BATCH]
            logger.info(f"Batch mode: processing first {pipeline.MAX_BATCH} annotation_id(s) for testing/demo purposes")
        for ann in unique_ids:
            _run_one(ann, use_file=False)

        # After batch run, compute & save aggregate metrics for the whole batch
        pipeline.evaluator.compute_metrics()
        batch_tag = f"batch_{os.path.basename(dataset_file_name)}" if dataset_file_name else "batch_all"
        saved = pipeline.evaluator.save_results(tag=batch_tag)
        logger.info(f"Batch evaluation saved: {saved}")

        # Prepare a consolidated result object
        result = {
            "success": all(r.get("success", False) for _, r in processed_results),
            "processed": [ann for ann, _ in processed_results],
            "evaluation": {
                "aggregate": pipeline.evaluator.aggregate.to_dict() if pipeline.evaluator.aggregate else {},
                "saved_files": saved,
            },
            "latency": sum(r.get("latency", 0.0) for _, r in processed_results),
        }

    # Display consolidated results (single or batch)
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)

    if result.get("success"):
        logger.info(f"\n✓ Pipeline completed successfully")
        eval_data = result.get("evaluation", {})
        if eval_data.get("saved_files"):
            logger.info(f"\nEvaluation files:")
            for kind, path in eval_data["saved_files"].items():
                logger.info(f"  {kind}: {path}")
    else:
        logger.error(f"\n✗ FAILED: {result.get('error', 'unknown')}")

    logger.info(f"\nLatency: {result.get('latency', 0.0):.2f}s")
    logger.info("\n" + "=" * 80)

    return result

