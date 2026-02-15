from AutoWeb.src.config import get_model_path
from AutoWeb.src.seeact_pipeline import run_single_prediction_example

if __name__ == "__main__":

    default_model_path = get_model_path()
    
    # Take one annotation ID for test for single task
    # we will keep default one annoation ID for test
    annoation_id = "401c4e6f-6b0b-47b4-8157-92d7ca468bbc" #Total 7 Steps in task
    dataset_file_name = "train-00000-of-00027-4d11798d7219186d.parquet"
    
    # Set use_gpt=True to use GPT-4o-mini (OpenAI API) instead of local Qwen2-VL-2B.
    # Requires OPENAI_API_KEY + SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY in .env
    run_single_prediction_example(
        model_folder= default_model_path,
        annotation_id=annoation_id,
        dataset_file_name=dataset_file_name,
        use_gpt=True,
    )
