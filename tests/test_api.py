import sys
import os
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
import base64
import sys

def test_pipeline(image_path):
    print(f"Testing API with image: {image_path}")
    
    url = "http://127.0.0.1:8000/detect"
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path, f, "image/jpeg")}
            response = requests.post(url, files=files)
            
        if response.status_code == 200:
            result = response.json()
            print("\n--- API Response ---")
            print(f"Final Sign:      {result['final_sign']}")
            print(f"Confidence:      {result['confidence']}")
            print(f"Critic Feedback: {result['critic_feedback']}")
            if result['drafter_guess'] != result['final_sign']:
                print(f"Drafter Guess:   {result['drafter_guess']}")
        else:
            print(f"Error {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n[Error] Could not connect to API. Is Uvicorn running?")
        print("Run: uvicorn api.main:app --reload")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <path_to_image>")
        # Create a red dummy image if none provided
        from PIL import Image
        import io
        img = Image.new('RGB', (224, 224), color = 'red')
        img.save("dummy_test.jpg")
        print("No image provided. Created and testing with 'dummy_test.jpg'")
        test_pipeline("dummy_test.jpg")
    else:
        test_pipeline(sys.argv[1])
