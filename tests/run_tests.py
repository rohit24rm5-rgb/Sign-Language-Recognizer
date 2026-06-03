import os
import sys
import glob
import random
import base64
import subprocess
import time

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ai_pipeline import SignDetectorPipeline

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data")
DATASET_NAME = "grassknoted/asl-alphabet"

def download_kaggle_dataset():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    print(f"Checking for dataset in {DATA_DIR}...")
    images = glob.glob(os.path.join(DATA_DIR, "**/*.jpg"), recursive=True)
    
    if len(images) < 20:
        print("Dataset not found or too small. Attempting to download via kaggle CLI...")
        try:
            # Ensure kaggle is installed
            import kaggle
            kaggle.api.authenticate()
            print("Downloading ASL dataset...")
            kaggle.api.dataset_download_files(DATASET_NAME, path=DATA_DIR, unzip=True)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please ensure your Kaggle API credentials are set up in ~/.kaggle/kaggle.json")
            print(f"Or manually download {DATASET_NAME} and place the images in the Data/ directory.")
            sys.exit(1)
            
    return glob.glob(os.path.join(DATA_DIR, "**/*.jpg"), recursive=True)

def run_automated_tests(num_tests=20):
    images = download_kaggle_dataset()
    if len(images) < num_tests:
        print(f"Not enough images found. Needed {num_tests}, found {len(images)}.")
        return
        
    # Pick random images
    test_images = random.sample(images, num_tests)
    pipeline = SignDetectorPipeline()
    
    success_count = 0
    total = 0
    
    print(f"\n--- Starting Automated Test Suite ({num_tests} Tests) ---")
    
    for img_path in test_images:
        total += 1
        # Use folder name as ground truth if using ASL alphabet, or manually inspect.
        # For this test, we just ensure the pipeline runs without crashing and check the critic's approval.
        print(f"Test {total}/{num_tests} - Image: {os.path.basename(img_path)}")
        
        with open(img_path, "rb") as f:
            base64_img = base64.b64encode(f.read()).decode('utf-8')
            
        try:
            result = pipeline.detect_sign(base64_img)
            print(f"  Drafter Guess: {result['drafter_guess']}")
            print(f"  Critic Approved: {result['critic_approved']}")
            print(f"  Final Sign: {result['final_sign']} (Confidence: {result['confidence']})")
            print(f"  Feedback: {result['critic_feedback']}\n")
            
            if result['critic_approved']:
                success_count += 1
                
        except Exception as e:
            print(f"  Error processing image: {e}\n")
            
        # Slight delay to respect API rate limits
        time.sleep(2)
        
    print(f"--- Test Complete ---")
    print(f"Critic Approval Rate: {success_count}/{total} ({(success_count/total)*100:.1f}%)")

if __name__ == "__main__":
    run_automated_tests()
