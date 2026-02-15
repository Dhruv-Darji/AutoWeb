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
from config import get_model_path, get_device, get_model_dtype, get_use_8bit, get_openai_model
from AutoWeb.src.model_interface import VLModel
from AutoWeb.src.gpt_model import GPTVisionModel
from AutoWeb.src.logger import logger


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
                 target_width: int = 1280,
                 target_height: int = 720,
                 device: Optional[str] = None,
                 use_gpt: bool = False):
        logger.info("[SeeActPipeline] Initializing components...")
        self.use_gpt = use_gpt
        
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
        self.action_grounding = SeeActActionGrounding(model=self.model)
        
        # 6. Evaluator (SeeAct offline metrics)
        self.evaluator = SeeActEvaluator(output_dir="eval_results")
        logger.info("  ✓ Evaluator ready")
        
        logger.info("[SeeActPipeline] ✓ Pipeline ready!")
    
    def predict_single_task(
            self, 
            annotation_id: str, 
            dataset_file_name: str,
            seeAct_method: str = "2" #1. Element Attributes 2. Textual Choice 3. Image Annotation
        ) -> Dict:
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
                history=action_history
            )
        
            # Step 3: Action Generation (take input from previous block, result a planning for task)
            logger.info("="*20 + " [3/6] Running Action Generation... " + "="*20)
            action_generation_plan = self.actionGenration.generate_plans(
                input_data=action_generation_input
            )

            output_plan = action_generation_plan.get("output_text", "")
            err = action_generation_plan.get("error")

            logger.info(f"Generated Action Plan: {output_plan}")

            if err:
                logger.error(f"✗ Action generation failed with error: {err}")
                # continue to next action or decide how to handle this case (e.g., skip grounding/decoding for this step)
                continue

            # store the input and action plan in history for potential use in future steps
            action_history.append({                
                "action_plan": output_plan
            })

            # Skip grounding if plan is empty — record as failed step directly
            if not output_plan.strip():
                logger.warning("    ⚠ Empty action plan — skipping grounding, recording as failed step.")
                grounding_result = {"success": False, "error": "empty plan", "selected_element": None}
            else:
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

def run_single_prediction_example(
    model_folder: str,
    annotation_id: Optional[str] = None,
    dataset_file_name: Optional[str] = None,
    device: Optional[str] = None,
    use_gpt: bool = False,
):
    """    
    # Pass the annoation ID and dataset file name to retrieve the screenshot and instruction for that particular annotation ID and then run the prediction on that.
    """
    backend_label = (
        f"{get_openai_model()} (OpenAI API)" if use_gpt else "Qwen2-VL-2B (local)"
    )
    logger.info("=" * 80)
    logger.info("SeeAct Single-Step UI Action Predictor")
    logger.info(f"  Model backend: {backend_label}")
    logger.info("=" * 80)
    
    # Initialize pipeline
    pipeline = SeeActPipeline(
        model_folder=model_folder,
        target_width=1280,
        target_height=720,
        use_gpt=use_gpt,
    )
    
    # Run prediction
    """THIS IS THE ACTUAL PREDICTION CALL FOR SINGLE TASK SEEACT"""
    result = pipeline.predict_single_task( annotation_id=annotation_id, dataset_file_name=dataset_file_name)
    
    # Display results
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
    
    logger.info(f"\nLatency: {result['latency']:.2f}s")
    logger.info("\n" + "=" * 80)
    
    return result

