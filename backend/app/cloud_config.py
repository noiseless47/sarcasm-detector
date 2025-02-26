"""Configuration for cloud storage and model access."""
import os
from google.cloud import storage
from pathlib import Path

# Google Cloud Storage configuration
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
if not GCS_BUCKET_NAME:
    raise ValueError("GCS_BUCKET_NAME environment variable is not set")

GCS_MODEL_FOLDER = "llama-models/Llama3.2-3B-Instruct"

# Local paths - using models folder directly
MODELS_DIR = Path(__file__).parent / "models"
LLAMA_MODEL_PATH = MODELS_DIR / "Llama3.2-3B-Instruct"

# Create models directory if it doesn't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LLAMA_MODEL_PATH.mkdir(parents=True, exist_ok=True)

def get_storage_client():
    """Get authenticated Google Cloud Storage client."""
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise ValueError(
            "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set. "
            "Please set it to the path of your service account key file."
        )
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(
            f"Credentials file not found at {credentials_path}"
        )
    
    try:
        return storage.Client()
    except Exception as e:
        print(f"Error initializing Google Cloud Storage client: {e}")
        print(f"Using credentials from: {credentials_path}")
        raise 