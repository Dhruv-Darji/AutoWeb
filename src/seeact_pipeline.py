"""
SeeAct-Style Single-Step UI Action Predictor Pipeline

This is the main pipeline that integrates all components:
1. Load dataset sample (image + instruction)
2. Preprocess image (minimal, SeeAct-style)
3. Build strict prompt
4. Run model inference
5. Decode and validate JSON output
6. Return structured ActionPrediction

STOPPING CRITERION:
"I can give a screenshot + instruction and my system outputs a valid JSON action prediction locally."
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from PIL import Image

# Add src to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from action_schema import ActionPrediction, ActionSchema
from action_decoder import ActionDecoder
from image_preprocessor import SeeActImagePreprocessor
from prompt_engine import PromptEngine
from model_interface import VLModel
from config import get_model_path, get_device, get_model_dtype, get_use_8bit


class SeeActPipeline:
    """
    Complete SeeAct-style single-step UI action prediction pipeline.
    
    This class integrates:
    - Image preprocessing (minimal, vision-only)
    - Prompt engineering (strict JSON output)
    - Model inference (local Qwen2-VL-2B)
    - JSON repair and validation
    - Action decoding
    
    Input: screenshot (PIL.Image) + instruction (str)
    Output: ActionPrediction (validated JSON)
    """
    
    def __init__(self,
                 model_folder: str,
                 target_width: int = 1280,
                 target_height: int = 720,
                 device: Optional[str] = None,
                 use_8bit: bool = False):
        """
        Initialize the SeeAct pipeline.
        
        Args:
            model_folder: Path to local Qwen2-VL-2B model
            target_width: Target image width for preprocessing
            target_height: Target image height for preprocessing
            device: "cuda" or "cpu" (None = auto)
            use_8bit: Whether to use 8-bit quantization
        """
        print("[SeeActPipeline] Initializing components...")
        
        # 1. Image Preprocessor (minimal, SeeAct-style)
        self.preprocessor = SeeActImagePreprocessor(
            target_width=target_width,
            target_height=target_height,
            keep_aspect_ratio=True,
            normalize=False  # Model handles normalization
        )
        print(f"  ✓ Image preprocessor ready ({target_width}x{target_height})")
        
        # 2. Prompt Engine (strict JSON enforcement)
        self.prompt_engine = PromptEngine()
        print("  ✓ Prompt engine ready (strict JSON mode)")
        
        # 3. Model Interface (local Qwen2-VL-2B)
        print(f"  ⏳ Loading model from {model_folder}...")
        self.model = VLModel(
            model_folder=model_folder,
            device=device,
            use_8bit=use_8bit
        )
        print("  ✓ Model loaded")
        
        # 4. Action Decoder (JSON repair + validation)
        self.decoder = ActionDecoder(strict_validation=False)
        print("  ✓ Action decoder ready")
        
        print("[SeeActPipeline] ✓ Pipeline ready!")
    
    def predict(self,
               image: Image.Image,
               instruction: str,
               max_new_tokens: int = 256,
               temperature: float = 0.0) -> Dict:
        """
        Predict a single UI action from screenshot + instruction.
        
        Args:
            image: PIL.Image of webpage screenshot
            instruction: Task instruction (e.g., "Click the login button")
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
        
        Returns:
            {
                "prediction": ActionPrediction or None,
                "raw_output": str (model raw text),
                "parsed_json": dict or None,
                "error": str or None,
                "success": bool,
                "latency": float (seconds),
                "scale_info": dict (for coordinate transformation)
            }
        """
        print(f"\n[SeeActPipeline] Predicting action for: '{instruction}'")
        
        # Step 1: Preprocess image
        print("  [1/5] Preprocessing image...")
        preprocessed = self.preprocessor.preprocess(image)
        processed_image = preprocessed["image"]
        scale_info = preprocessed["scale_info"]
        
        # Update decoder with image dimensions for coordinate normalization
        self.decoder.set_image_dimensions(
            scale_info["new_width"],
            scale_info["new_height"]
        )
        
        print(f"        Original: {scale_info['original_width']}x{scale_info['original_height']}")
        print(f"        Resized:  {scale_info['new_width']}x{scale_info['new_height']}")
        print(f"        Scale:    {scale_info['scale_x']:.3f}")
        
        # Step 2: Build prompt
        print("  [2/5] Building prompt...")
        # Ask model/processor what image placeholder (if any) it prefers so tokenization and
        # when the processor prefers implicit image tokens.
        try:
            image_key = self.model.get_preferred_image_key(processed_image)
            if image_key == "":
                print("        Using implicit image tokens (no explicit placeholder)")
            else:
                print(f"        Using image placeholder: {image_key!r}")
        except Exception as e:
            image_key = "<image>"
            print(f"        ⚠ could not detect image placeholder (falling back to '<image>'): {e}")

        prompt_text, _ = self.prompt_engine.build_prompt(
            task_text=instruction,
            variant="strict_json",
            include_dom=False,  # SeeAct: no DOM, vision-only
            image_key=image_key or "<image>"
        )
        
        # Step 3: Run model inference
        print("  [3/5] Running model inference...")
        inference_result = self.model.infer(
            image_or_tensor=processed_image,
            prompt_text=prompt_text,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01)  # Avoid temp=0 issues
        )
        
        raw_output = inference_result["raw_text"]
        latency = inference_result["latency"]
        
        print(f"        Latency: {latency:.2f}s")
        print(f"        Raw output (first 100 chars): {raw_output[:100]}...")
        
        # Step 4: Decode and repair JSON
        print("  [4/5] Decoding action...")
        decode_result = self.decoder.decode_with_metadata(raw_output)
        
        prediction = decode_result["prediction"]
        parsed_json = decode_result["parsed_json"]
        error = decode_result["error"]
        success = decode_result["success"]
        
        if success:
            print(f"        ✓ Action: {prediction.action_type}")
            print(f"        ✓ Confidence: {prediction.confidence:.2f}")
        else:
            print(f"        ✗ Decode failed: {error}")
        
        # Step 5: Validate schema
        if prediction:
            print("  [5/5] Validating schema...")
            is_valid, validation_error = ActionSchema.validate_prediction(prediction)
            if is_valid:
                print("        ✓ Schema valid")
            else:
                print(f"        ⚠ Schema validation warning: {validation_error}")
        else:
            print("  [5/5] Skipping validation (no prediction)")
        
        return {
            "prediction": prediction,
            "raw_output": raw_output,
            "parsed_json": parsed_json,
            "error": error,
            "success": success,
            "latency": latency,
            "scale_info": scale_info,
            "prompt": prompt_text  # Include for debugging
        }
    
    def predict_from_path(self,
                         image_path: str,
                         instruction: str,
                         **kwargs) -> Dict:
        """
        Convenience method to predict from image file path.
        
        Args:
            image_path: Path to image file
            instruction: Task instruction
            **kwargs: Additional arguments for predict()
        
        Returns:
            Prediction result dict (same as predict())
        """
        image = Image.open(image_path).convert("RGB")
        return self.predict(image, instruction, **kwargs)


def run_single_prediction_example(
    model_folder: str,
    image_path: str,
    instruction: str,
    device: Optional[str] = None,
    use_8bit: bool = False
):
    """
    Example: Run a single prediction.
    
    This demonstrates the STOPPING CRITERION:
    "I can give a screenshot + instruction and my system outputs a valid JSON action prediction locally."
    
    Args:
        model_folder: Path to model directory
        image_path: Path to screenshot image
        instruction: Task instruction
        device: Device to use ("cuda", "cpu", or None for auto)
        use_8bit: Whether to use 8-bit quantization
    """
    print("=" * 80)
    print("SeeAct Single-Step UI Action Predictor")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = SeeActPipeline(
        model_folder=model_folder,
        target_width=1280,
        target_height=720,
        device=device,
        use_8bit=use_8bit
    )
    
    # Run prediction
    result = pipeline.predict_from_path(image_path, instruction)
    
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


if __name__ == "__main__":
    # Example usage
    import argparse
    from pathlib import Path
    
    # Get default model path from .env
    default_model_path = get_model_path()
    
    parser = argparse.ArgumentParser(
        description="Run SeeAct single-step prediction"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a lightweight bundled demo (creates a placeholder screenshot and example instruction)"
    )
    
    args = parser.parse_args()
    
    # CLI validation / demo fallback
    if args.demo:
        # create a lightweight placeholder image next to the script for reproducible demo runs
        demo_path = Path(__file__).parent / "demo_screenshot.jpg"
        if not demo_path.exists():
            from PIL import Image
            Image.new("RGB", (1280, 720), color=(240, 240, 240)).save(demo_path)
        image_path = str(demo_path)
        instruction = "Click the login button"
    else:
        # Temporary Default demo added
        demo_path = Path(__file__).parent / "demo_screenshot.jpg"
        if not demo_path.exists():
            from PIL import Image
            Image.new("RGB", (1280, 720), color=(240, 240, 240)).save(demo_path)
        image_path = str(demo_path)
        instruction = "Click the login button"
        # parser.error(
        #     "--demomust be provided.\n"
        #     "Example (Windows PowerShell):\n"
        #     "  python .\\seeact_pipeline.py --demo\n"
        # )
    
    run_single_prediction_example(
        model_folder= default_model_path,
        image_path=image_path,
        instruction=instruction
    )
