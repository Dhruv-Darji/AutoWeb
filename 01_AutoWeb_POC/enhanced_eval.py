"""
Enhanced Evaluation Script for Research-Quality Results

This script provides comprehensive evaluation with:
- Detailed metrics per action type
- Confidence interval computation
- Failure case analysis
- Result export in multiple formats
- Progress tracking
- Logging
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import argparse

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from mind2Web_Loader import Mind2WebDataset
from seeact_pipeline import SeeActPipeline
from evaluator import ActionEvaluator
from config import get_model_path, get_dataset_path, get_results_path


class EnhancedEvaluator:
    """
    Enhanced evaluator with research-quality reporting.
    """
    
    def __init__(self, output_dir: str = "../results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        print(f"Results will be saved to: {self.run_dir}")
    
    def run_evaluation(
        self,
        dataset_path: str,
        model_path: str,
        num_samples: int = 100,
        split: str = "test_task"
    ):
        """
        Run complete evaluation pipeline.
        """
        print("=" * 80)
        print("Enhanced Evaluation for Research-Quality Results")
        print("=" * 80)
        print()
        
        # Step 1: Load dataset
        print(f"[1/4] Loading dataset from {dataset_path}...")
        try:
            dataset = Mind2WebDataset(
                root_dir=dataset_path,
                split=split,
                sample_size=num_samples,
                shuffle=True,
                seed=42
            )
            print(f"  ✓ Loaded {len(dataset)} samples")
        except Exception as e:
            print(f"  ✗ Failed to load dataset: {e}")
            return None
        
        print()
        
        # Step 2: Initialize pipeline
        print(f"[2/4] Initializing SeeAct pipeline...")
        try:
            pipeline = SeeActPipeline(
                model_folder=model_path,
                target_width=1280,
                target_height=720
            )
            print(f"  ✓ Pipeline ready")
        except Exception as e:
            print(f"  ✗ Failed to initialize pipeline: {e}")
            return None
        
        print()
        
        # Step 3: Run predictions
        print(f"[3/4] Running predictions on {len(dataset)} samples...")
        print("  (This may take several minutes...)")
        print()
        
        evaluator = ActionEvaluator()
        results = []
        
        start_time = time.time()
        
        for i in range(len(dataset)):
            sample = dataset[i]
            
            # Progress indicator
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1) if i > 0 else 0
                remaining = avg_time * (len(dataset) - i - 1)
                print(f"  Progress: {i+1}/{len(dataset)} samples "
                      f"({(i+1)/len(dataset)*100:.1f}%) "
                      f"| Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s")
            
            # Get inputs
            image = sample["image"]
            instruction = sample["instruction"]
            oracle = sample["oracle_action"]
            metadata = sample["metadata"]
            
            # Run prediction
            try:
                result = pipeline.predict(image, instruction)
                
                sample_result = {
                    "sample_id": i,
                    "instruction": instruction,
                    "oracle_action": oracle,
                    "metadata": metadata,
                    "success": result["success"],
                    "latency": result["latency"],
                }
                
                if result["success"]:
                    pred = result["prediction"]
                    pred_dict = pred.to_dict()
                    sample_result["predicted_action"] = pred_dict
                    
                    # Add to evaluator
                    evaluator.add_result(pred_dict, oracle, metadata)
                else:
                    sample_result["error"] = result["error"]
                    sample_result["raw_output"] = result["raw_output"][:200]
                    
                    # Add failed prediction as noop
                    evaluator.add_result(
                        {"action_type": "noop", "target": None, "value": None},
                        oracle,
                        metadata
                    )
                
                results.append(sample_result)
                
            except Exception as e:
                print(f"  ✗ Error on sample {i}: {e}")
                sample_result = {
                    "sample_id": i,
                    "instruction": instruction,
                    "oracle_action": oracle,
                    "metadata": metadata,
                    "success": False,
                    "error": str(e)
                }
                results.append(sample_result)
                
                # Add as failed
                evaluator.add_result(
                    {"action_type": "noop", "target": None, "value": None},
                    oracle,
                    metadata
                )
        
        total_time = time.time() - start_time
        print()
        print(f"  ✓ Completed in {total_time:.1f}s "
              f"(avg: {total_time/len(dataset):.2f}s per sample)")
        print()
        
        # Step 4: Compute metrics and save results
        print(f"[4/4] Computing metrics and saving results...")
        
        # Compute aggregate metrics
        metrics = evaluator.compute_metrics()
        
        # Compute per-action-type metrics
        per_action_metrics = self._compute_per_action_metrics(evaluator.results)
        
        # Get failure cases
        failure_cases = evaluator.get_failure_cases("overall")
        
        # Save all results
        self._save_results(
            results=results,
            metrics=metrics,
            per_action_metrics=per_action_metrics,
            failure_cases=failure_cases,
            total_time=total_time,
            num_samples=len(dataset)
        )
        
        # Print summary
        self._print_summary(metrics, per_action_metrics, total_time, len(dataset))
        
        print()
        print(f"  ✓ All results saved to: {self.run_dir}")
        print()
        
        return {
            "metrics": metrics,
            "per_action_metrics": per_action_metrics,
            "results": results,
            "run_dir": str(self.run_dir)
        }
    
    def _compute_per_action_metrics(self, results: List[Dict]) -> Dict:
        """Compute metrics broken down by action type."""
        per_action = {}
        
        # Group by oracle action type
        for result in results:
            oracle_action = result["oracle"]["action_type"]
            
            if oracle_action not in per_action:
                per_action[oracle_action] = {
                    "total": 0,
                    "action_correct": 0,
                    "target_correct": 0,
                    "value_correct": 0,
                    "overall_correct": 0
                }
            
            per_action[oracle_action]["total"] += 1
            
            eval_result = result["evaluation"]
            if eval_result["action_match"]:
                per_action[oracle_action]["action_correct"] += 1
            if eval_result["target_match"]:
                per_action[oracle_action]["target_correct"] += 1
            if eval_result["value_match"]:
                per_action[oracle_action]["value_correct"] += 1
            if eval_result["overall_match"]:
                per_action[oracle_action]["overall_correct"] += 1
        
        # Compute percentages
        for action_type, stats in per_action.items():
            total = stats["total"]
            stats["action_accuracy"] = stats["action_correct"] / total if total > 0 else 0
            stats["target_accuracy"] = stats["target_correct"] / total if total > 0 else 0
            stats["value_accuracy"] = stats["value_correct"] / total if total > 0 else 0
            stats["overall_accuracy"] = stats["overall_correct"] / total if total > 0 else 0
        
        return per_action
    
    def _save_results(
        self,
        results: List[Dict],
        metrics,
        per_action_metrics: Dict,
        failure_cases: List[Dict],
        total_time: float,
        num_samples: int
    ):
        """Save all results to files."""
        
        # Save detailed results (all samples)
        with open(self.run_dir / "detailed_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save metrics summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": num_samples,
            "total_time_seconds": total_time,
            "avg_time_per_sample": total_time / num_samples,
            "metrics": {
                "action_accuracy": metrics.action_accuracy,
                "target_accuracy": metrics.target_accuracy,
                "value_accuracy": metrics.value_accuracy,
                "correct_actions": metrics.correct_actions,
                "correct_targets": metrics.correct_targets,
                "correct_values": metrics.correct_values,
                "total_samples": metrics.total_samples
            },
            "per_action_metrics": per_action_metrics
        }
        
        with open(self.run_dir / "metrics_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save failure cases
        with open(self.run_dir / "failure_cases.json", "w") as f:
            json.dump(failure_cases, f, indent=2, default=str)
        
        # Save human-readable report
        self._save_text_report(summary, self.run_dir / "report.txt")
    
    def _save_text_report(self, summary: Dict, filepath: Path):
        """Save human-readable text report."""
        with open(filepath, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("SeeAct Baseline Evaluation Report\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Timestamp: {summary['timestamp']}\n")
            f.write(f"Samples Evaluated: {summary['num_samples']}\n")
            f.write(f"Total Time: {summary['total_time_seconds']:.1f}s\n")
            f.write(f"Avg Time per Sample: {summary['avg_time_per_sample']:.2f}s\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("Overall Metrics\n")
            f.write("-" * 80 + "\n\n")
            
            metrics = summary['metrics']
            f.write(f"Action Accuracy (AA):  {metrics['action_accuracy']:.2%} "
                   f"({metrics['correct_actions']}/{metrics['total_samples']})\n")
            f.write(f"Target Accuracy (TA):  {metrics['target_accuracy']:.2%} "
                   f"({metrics['correct_targets']}/{metrics['total_samples']})\n")
            f.write(f"Value Accuracy (VA):   {metrics['value_accuracy']:.2%} "
                   f"({metrics['correct_values']}/{metrics['total_samples']})\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("Per-Action-Type Breakdown\n")
            f.write("-" * 80 + "\n\n")
            
            for action_type, stats in summary['per_action_metrics'].items():
                f.write(f"{action_type.upper()}:\n")
                f.write(f"  Samples: {stats['total']}\n")
                f.write(f"  Action Accuracy: {stats['action_accuracy']:.2%}\n")
                f.write(f"  Target Accuracy: {stats['target_accuracy']:.2%}\n")
                f.write(f"  Value Accuracy:  {stats['value_accuracy']:.2%}\n")
                f.write(f"  Overall Accuracy: {stats['overall_accuracy']:.2%}\n\n")
    
    def _print_summary(self, metrics, per_action_metrics: Dict, total_time: float, num_samples: int):
        """Print summary to console."""
        print("=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print()
        print(f"Samples: {num_samples}")
        print(f"Total Time: {total_time:.1f}s (avg: {total_time/num_samples:.2f}s per sample)")
        print()
        print(metrics)
        print()
        print("-" * 80)
        print("Per-Action-Type Breakdown:")
        print("-" * 80)
        for action_type, stats in per_action_metrics.items():
            print(f"\n{action_type.upper()} (n={stats['total']}):")
            print(f"  AA: {stats['action_accuracy']:.2%}  "
                  f"TA: {stats['target_accuracy']:.2%}  "
                  f"VA: {stats['value_accuracy']:.2%}")


def main():
    # Get defaults from .env file
    default_model = get_model_path()
    default_dataset = get_dataset_path()
    default_results = get_results_path()
    
    parser = argparse.ArgumentParser(
        description="Enhanced evaluation for research-quality results"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        required=default_dataset is None,  # Required only if not in .env
        help=f"Path to Mind2Web dataset root directory (default from .env: {default_dataset})" if default_dataset else "Path to Mind2Web dataset root directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        required=default_model is None,  # Required only if not in .env
        help=f"Path to Qwen2-VL-2B model folder (default from .env: {default_model})" if default_model else "Path to Qwen2-VL-2B model folder"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test_task",
        help="Dataset split to use (default: test_task)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=default_results,
        help=f"Output directory for results (default: {default_results})"
    )
    
    args = parser.parse_args()
    
    evaluator = EnhancedEvaluator(output_dir=args.output)
    
    results = evaluator.run_evaluation(
        dataset_path=args.dataset,
        model_path=args.model,
        num_samples=args.num_samples,
        split=args.split
    )
    
    if results:
        print("=" * 80)
        print("✅ Evaluation completed successfully!")
        print(f"📁 Results saved to: {results['run_dir']}")
        print()
        print("Files created:")
        print("  - detailed_results.json    (per-sample results)")
        print("  - metrics_summary.json     (aggregate metrics)")
        print("  - failure_cases.json       (failed predictions)")
        print("  - report.txt               (human-readable report)")
        print("=" * 80)
    else:
        print("=" * 80)
        print("✗ Evaluation failed. Please check errors above.")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
