"""
Final plan available at AutoWeb/Docs/SeeAct_understanding.md
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from PIL import Image
from transformers import pipeline

# Add src to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from action_schema import ActionPrediction, ActionSchema
from action_decoder import ActionDecoder
from image_preprocessor import SeeActImagePreprocessor
from prompt_engine import PromptEngine
from model_interface import VLModel
from config import get_model_path, get_device, get_model_dtype, get_use_8bit


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
                 use_8bit: bool = False):
        print("[SeeActPipeline] Initializing components...")
        
        # 1. Mind2Web Dataset Loader (for retrieving task data based on annotation ID)
        self.mind2web_loader = Mind2WebDataset(root_dir="D:\\Environments\\Datasets\\multimodal-mind2web")
        print(f"  ✓ Mind2Web dataset loader ready(root: D:\\Environments\\Datasets\\multimodal-mind2web)")
        
        # 2. Prompt Engine (strict JSON enforcement)
        self.prompt_engine = PromptEngine()
        print("  ✓ Prompt engine ready (strict JSON mode)")
        
        # 3. Model Interface (local Qwen2-VL-2B)
        print(f"  ⏳ Loading model from {model_folder}...")
        # self.model = VLModel(
        #     model_folder=model_folder,
        #     device=device,
        #     use_8bit=use_8bit
        # )
        print("  ✓ Model loaded")
        
        # 4. Action Decoder (JSON repair + validation)
        self.decoder = ActionDecoder(strict_validation=False)
        print("  ✓ Action decoder ready")
        
        print("[SeeActPipeline] ✓ Pipeline ready!")
    
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

        print(f"\n[SeeActPipeline] Predicting action for: '{annotation_id}'")
        
        # Step 1: Load single task data (multiple steps (with details like screenshot, cleaned HTML) + instruction)
        print("="*20 , "[1/6] Loading task data...", "="*20)
        single_task = self.mind2web_loader.get_task(
            annotation_id=annotation_id,
            file_name=dataset_file_name
        )
        
        print(f"Instruction: {single_task[0]['instruction']}")

        # # Update decoder with image dimensions for coordinate normalization
        # self.decoder.set_image_dimensions(
        #     scale_info["new_width"],
        #     scale_info["new_height"]
        # )
        
        # print(f"        Original: {scale_info['original_width']}x{scale_info['original_height']}")
        # print(f"        Resized:  {scale_info['new_width']}x{scale_info['new_height']}")
        # print(f"        Scale:    {scale_info['scale_x']:.3f}")
        
        # for each step in task_steps:

        # Step 2: Prepare inputs for single task (stepImage + past history + instruction)
        print("="*20 , "[2/6] Input Preparation...", "="*20)
        # Ask model/processor what image placeholder (if any) it prefers so tokenization and
        # when the processor prefers implicit image tokens.
        
        # Step 3: Action Generation (take input from previous block, result a planning for task)
        print("="*20 , "[3/6] Running Action Generation...", "="*20)

        # prompt_text, _ = self.prompt_engine.build_prompt(
        #     task_text=instruction,
        #     variant="strict_json",
        #     include_dom=False,  # SeeAct: no DOM, vision-only
        #     image_key=image_key or "<image>"
        # )
        
        # Step 4: Grounding method selection and processing
        print("="*20 , "[4/6] Running Action Grounding ...", "="*20)
        # self.action_grounding.process(method=seeAct_method, annotation_id=annotation_id, dataset_file_name=dataset_file_name )
                        
        
        # raw_output = inference_result["raw_text"]
        # latency = inference_result["latency"]
        
        # print(f"        Latency: {latency:.2f}s")
        # print(f"        Raw output (first 100 chars): {raw_output[:100]}...")
        
        # Step 5: Parse the Grounding output 
        print("="*20 , "[5/6] Parsing output...", "="*20)
        # decode_result = self.decoder.decode_with_metadata(raw_output)
        
        # prediction = decode_result["prediction"]
        # parsed_json = decode_result["parsed_json"]
        # error = decode_result["error"]
        # success = decode_result["success"]
        
        # if success:
        #     print(f"        ✓ Action: {prediction.action_type}")
        #     print(f"        ✓ Confidence: {prediction.confidence:.2f}")
        # else:
        #     print(f"        ✗ Decode failed: {error}")
        
        # # Step 6: Evaluate
        print("="*20 , "[6/6] Evaluating prediction...", "="*20)
        # if prediction:
        #     print("  [6/6] Validating schema...")
        #     is_valid, validation_error = ActionSchema.validate_prediction(prediction)
        #     if is_valid:
        #         print("        ✓ Schema valid")
        #     else:
        #         print(f"        ⚠ Schema validation warning: {validation_error}")
        # else:
        #     print("  [6/6 ] Skipping validation (no prediction)")
        
        # return {
        #     "prediction": prediction,
        #     "raw_output": raw_output,
        #     "parsed_json": parsed_json,
        #     "error": error,
        #     "success": success,
        #     "latency": latency,
        #     "prompt": prompt_text  # Include for debugging
        # }  

        return {
            "success": False,
            "error": "Not implemented yet",
            "latency": 0.0,
            "raw_output": "",
            "prediction": None            
            }  

def run_single_prediction_example(
    model_folder: str,
    annotation_id: Optional[str] = None,
    dataset_file_name: Optional[str] = None,
    device: Optional[str] = None,
):
    """    
    # Pass the annoation ID and dataset file name to retrieve the screenshot and instruction for that particular annotation ID and then run the prediction on that.
    """
    print("=" * 80)
    print("SeeAct Single-Step UI Action Predictor")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = SeeActPipeline(
        model_folder=model_folder,
        target_width=1280,
        target_height=720,
    )
    
    # Run prediction
    """THIS IS THE ACTUAL PREDICTION CALL FOR SINGLE TASK SEEACT"""
    result = pipeline.predict_single_task( annotation_id=annotation_id, dataset_file_name=dataset_file_name)
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    if result["success"]:
        pred = result["prediction"]
        print(f"\n✓ SUCCESS: Valid action prediction generated!")
        print(f"\nPredicted Action:")
        print(pred.to_json())
    else:
        print(f"\n✗ FAILED: {result['error']}")
        print(f"\nRaw model output:")
        print(result["raw_output"])
    
    print(f"\nLatency: {result['latency']:.2f}s")
    print("\n" + "=" * 80)
    
    return result

