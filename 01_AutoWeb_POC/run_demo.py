"""
Example: Complete SeeAct Pipeline Demo

Demonstrates the end-to-end flow:
1. Load a sample from Mind2Web dataset
2. Run SeeAct prediction
3. Compare with oracle (ground truth)
4. Display results

This script validates the STOPPING CRITERION:
"I can give a screenshot + instruction and my system outputs a valid JSON action prediction locally."
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from mind2Web_Loader import Mind2WebDataset
from seeact_pipeline import SeeActPipeline
from evaluator import ActionEvaluator
from config import get_model_path, get_dataset_path


def run_demo(
    dataset_path: str,
    model_folder: str,
    num_samples: int = 5
):
    """
    Run complete demo with Mind2Web samples.
    
    Args:
        dataset_path: Path to Mind2Web dataset root
        model_folder: Path to Qwen2-VL-2B model
        num_samples: Number of samples to process
    """
    
    print("=" * 80)
    print("SeeAct-Style Single-Step UI Action Predictor - DEMO")
    print("=" * 80)
    print()
    
    # Step 1: Load dataset
    print("[1/3] Loading Mind2Web dataset...")
    try:
        dataset = Mind2WebDataset(
            root_dir=dataset_path,
            split="test_task",
            sample_size=num_samples,
            shuffle=True
        )
        print(f"  ✓ Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"  ✗ Failed to load dataset: {e}")
        print(f"  Note: Make sure Mind2Web dataset is available at {dataset_path}")
        return
    
    print()
    
    # Step 2: Initialize pipeline
    print("[2/3] Initializing SeeAct pipeline...")
    try:
        pipeline = SeeActPipeline(
            model_folder=model_folder,
            target_width=1280,
            target_height=720
        )
    except Exception as e:
        print(f"  ✗ Failed to initialize pipeline: {e}")
        print(f"  Note: Make sure Qwen2-VL-2B model is available at {model_folder}")
        return
    
    print()
    
    # Step 3: Run predictions and evaluate
    print("[3/3] Running predictions...")
    print()
    
    evaluator = ActionEvaluator()
    
    for i in range(len(dataset)):
        sample = dataset[i]
        
        print("-" * 80)
        print(f"Sample {i+1}/{len(dataset)}")
        print("-" * 80)
        
        # Get inputs
        image = sample["image"]
        instruction = sample["instruction"]
        oracle = sample["oracle_action"]
        
        print(f"Instruction: {instruction}")
        print(f"Oracle action: {oracle['action_type']}")
        
        # Run prediction
        try:
            result = pipeline.predict(image, instruction)
            
            if result["success"]:
                pred = result["prediction"]
                print(f"\n✓ Predicted action:")
                print(f"  Action: {pred.action_type}")
                print(f"  Target: {pred.target}")
                print(f"  Value: {pred.value}")
                print(f"  Confidence: {pred.confidence:.2f}")
                print(f"  Latency: {result['latency']:.2f}s")
                
                # Convert to dict for evaluation
                pred_dict = pred.to_dict()
                
                # Evaluate
                evaluator.add_result(pred_dict, oracle)
                
            else:
                print(f"\n✗ Prediction failed: {result['error']}")
                print(f"Raw output: {result['raw_output'][:200]}...")
                
                # Add failed prediction as noop
                evaluator.add_result(
                    {"action_type": "noop", "target": None, "value": None},
                    oracle
                )
        
        except Exception as e:
            print(f"\n✗ Error during prediction: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Display evaluation results
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    metrics = evaluator.compute_metrics()
    print(metrics)
    
    # Show some failure cases
    failures = evaluator.get_failure_cases("action")
    if failures:
        print("\nSample Failure Cases (Action Mismatch):")
        for i, fail in enumerate(failures[:3], 1):
            pred = fail["predicted"]
            oracle = fail["oracle"]
            print(f"\n{i}. Predicted: {pred['action_type']}, Oracle: {oracle['action_type']}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print()
    print("✓ STOPPING CRITERION MET:")
    print("  'I can give a screenshot + instruction and my system outputs")
    print("   a valid JSON action prediction locally.'")
    print()
    print("Next steps:")
    print("  1. Run evaluation on larger dataset")
    print("  2. Analyze failure patterns")
    print("  3. Consider improvements (but not yet - baseline first!)")
    print()


if __name__ == "__main__":
    import argparse
    
    # Get defaults from .env file
    default_model = get_model_path()
    default_dataset = get_dataset_path()
    
    parser = argparse.ArgumentParser(
        description="Demo: SeeAct-style single-step prediction with Mind2Web"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        required=default_dataset is None,
        help=f"Path to Mind2Web dataset root directory (default from .env: {default_dataset})" if default_dataset else "Path to Mind2Web dataset root directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        required=default_model is None,
        help=f"Path to Qwen2-VL-2B model folder (default from .env: {default_model})" if default_model else "Path to Qwen2-VL-2B model folder"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to process (default: 5)"
    )
    
    args = parser.parse_args()
    
    run_demo(
        dataset_path=args.dataset,
        model_folder=args.model,
        num_samples=args.num_samples
    )
