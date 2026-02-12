from src.config import get_model_path
from src.seeact_pipeline import run_single_prediction_example

import argparse
from pathlib import Path

if __name__ == "__main__":    
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
