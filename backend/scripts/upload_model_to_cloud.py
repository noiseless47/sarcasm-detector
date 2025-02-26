"""Upload the LLaMA model to Google Cloud Storage."""
import os
import sys
from pathlib import Path
from google.cloud import storage
from google.api_core import retry

def upload_model():
    """Upload model files to Google Cloud Storage."""
    print("Starting model upload to Google Cloud Storage...")
    
    # Get bucket name from environment
    bucket_name = os.environ.get("GCS_BUCKET_NAME", "text-analyser-bucket")
    
    # Source directory (where your model files are currently located)
    source_dir = Path(__file__).parent.parent / "app" / "models" / "Llama3.2-3B-Instruct"
    
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return 1
        
    # List files to upload
    files = list(source_dir.glob("*"))
    print("\nFiles found to upload:")
    for f in files:
        print(f"- {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
    
    # Confirm upload
    response = input("\nDo you want to upload these files? (y/n): ")
    if response.lower() != 'y':
        print("Upload cancelled")
        return 1
    
    try:
        # Initialize storage client with increased timeout
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Upload each file
        for file_path in files:
            destination_blob_name = f"llama-models/Llama3.2-3B-Instruct/{file_path.name}"
            
            # For large files, use chunked upload
            file_size = os.path.getsize(file_path)
            
            if file_size > 100 * 1024 * 1024:  # If file is larger than 100MB
                print(f"\nUploading {file_path.name}...")
                print(f"Large file detected ({file_size / 1024 / 1024:.2f} MB). Using chunked upload...")
                
                # Create blob with chunk size
                blob = bucket.blob(
                    destination_blob_name,
                    chunk_size=256 * 1024 * 1024  # 256MB chunks
                )
                
                # Upload with chunked approach
                with open(file_path, 'rb') as f:
                    blob.upload_from_file(
                        f,
                        content_type='application/octet-stream',
                        retry=retry.Retry(deadline=3600),  # 1 hour timeout
                        timeout=3600  # 1 hour timeout
                    )
            else:
                # For small files, use simple upload
                print(f"\nUploading {file_path.name}...")
                blob = bucket.blob(destination_blob_name)
                blob.upload_from_filename(str(file_path))
            
            print(f"✓ Uploaded {file_path.name}")
        
        print("\nModel upload completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error uploading files: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(upload_model()) 