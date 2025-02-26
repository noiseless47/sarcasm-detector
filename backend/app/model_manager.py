"""Manager for model files in cloud storage."""
import os
import json
from pathlib import Path
from google.cloud import storage
from .cloud_config import (
    GCS_BUCKET_NAME, 
    GCS_MODEL_FOLDER,
    LLAMA_MODEL_PATH,
    get_storage_client
)

class ModelManager:
    """Manages model files between cloud storage and local cache."""
    
    def __init__(self):
        """Initialize the model manager."""
        self.client = get_storage_client()
        self.bucket = self.client.bucket(GCS_BUCKET_NAME)
        print(f"ModelManager initialized with bucket: {GCS_BUCKET_NAME}")
    
    def download_model_if_needed(self, force_download=False):
        """Download model files if they don't exist locally or if force_download is True."""
        required_files = [
            "params.json",
            "tokenizer.model",
            "consolidated.00.pth"
        ]
        
        # Check if files already exist locally
        all_files_exist = all(
            (LLAMA_MODEL_PATH / file).exists() 
            for file in required_files
        )
        
        if all_files_exist and not force_download:
            print("All model files already exist locally")
            return True
        
        print(f"Downloading model files from gs://{GCS_BUCKET_NAME}/{GCS_MODEL_FOLDER}/")
        
        try:
            # Download all required files
            for file in required_files:
                self._download_blob(
                    f"{GCS_MODEL_FOLDER}/{file}",
                    LLAMA_MODEL_PATH / file
                )
            
            print("All model files downloaded successfully")
            return True
        except Exception as e:
            print(f"Error downloading model files: {e}")
            return False
    
    def _download_blob(self, source_blob_name, destination_file):
        """Download a single blob from GCS bucket."""
        blob = self.bucket.blob(source_blob_name)
        
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(destination_file), exist_ok=True)
        
        print(f"Downloading {source_blob_name} to {destination_file}")
        blob.download_to_filename(destination_file)
    
    def upload_model(self):
        """Upload model files to cloud storage."""
        try:
            for file_path in LLAMA_MODEL_PATH.glob("**/*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(LLAMA_MODEL_PATH.parent)
                    destination_blob_name = f"{GCS_MODEL_FOLDER}/{file_path.name}"
                    
                    blob = self.bucket.blob(destination_blob_name)
                    blob.upload_from_filename(str(file_path))
                    print(f"Uploaded {file_path} to {destination_blob_name}")
            
            print("Model upload complete")
            return True
        except Exception as e:
            print(f"Error uploading model: {e}")
            return False
    
    def get_model_path(self):
        """Get the path to the local model files."""
        # Ensure model is downloaded
        self.download_model_if_needed()
        return LLAMA_MODEL_PATH 

    def upload_model_from_dir(self, source_dir):
        """Upload model files from a directory to cloud storage."""
        try:
            source_dir = Path(source_dir)
            print(f"\nUploading files from {source_dir} to gs://{GCS_BUCKET_NAME}/{GCS_MODEL_FOLDER}/")
            
            for file_path in source_dir.glob("*"):
                if file_path.is_file():
                    destination_blob_name = f"{GCS_MODEL_FOLDER}/{file_path.name}"
                    print(f"\nUploading {file_path.name}...")
                    
                    blob = self.bucket.blob(destination_blob_name)
                    blob.upload_from_filename(str(file_path))
                    print(f"✓ Uploaded {file_path.name}")
            
            print("\nModel upload complete")
            return True
        except Exception as e:
            print(f"Error uploading model: {e}")
            return False 