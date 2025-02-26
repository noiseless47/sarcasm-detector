import os
from google.cloud import storage
from pathlib import Path

def test_gcs_permissions():
    print("Testing Google Cloud Storage permissions...")
    
    # Get bucket name from environment
    bucket_name = os.environ.get("GCS_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("GCS_BUCKET_NAME environment variable not set")
    
    print(f"Using bucket: {bucket_name}")
    
    # Initialize client
    client = storage.Client()
    
    # Test bucket access
    try:
        bucket = client.get_bucket(bucket_name)
        print("✓ Successfully accessed bucket")
    except Exception as e:
        print(f"✗ Failed to access bucket: {e}")
        return False
    
    # Test object creation
    try:
        blob = bucket.blob("test.txt")
        blob.upload_from_string("test content")
        print("✓ Successfully created test object")
        
        # Clean up
        blob.delete()
        print("✓ Successfully deleted test object")
    except Exception as e:
        print(f"✗ Failed to create/delete object: {e}")
        return False
    
    print("All permission tests passed!")
    return True

if __name__ == "__main__":
    test_gcs_permissions() 