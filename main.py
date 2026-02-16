from AutoWeb.src.config import get_model_path
from AutoWeb.src.seeact_pipeline import run_single_prediction_example

if __name__ == "__main__":

    default_model_path = get_model_path()
    
    # To run a single task: set `annotation_id` to that task's id (example below).
    # To run the whole file (every unique annotation_id in the parquet), set
    # `annotation_id = None` and provide `dataset_file_name`.

    # Example: single-task run (default)
    annotation_id = "401c4e6f-6b0b-47b4-8157-92d7ca468bbc"

    # Example: batch run (process all unique annotation_id in the parquet)
    # annotation_id = None

    dataset_file_name = "train-00000-of-00027-4d11798d7219186d.parquet"

    # Set use_gpt=True to use GPT-4o-mini (OpenAI API) instead of local Qwen2-VL-2B.
    # Requires OPENAI_API_KEY + SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY in .env
    run_single_prediction_example(
        model_folder= default_model_path,
        annotation_id=annotation_id,
        dataset_file_name=dataset_file_name,
        use_gpt=True,
    )
