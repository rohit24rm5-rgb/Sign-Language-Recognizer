import sys
import os
import asyncio
import base64

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ai_pipeline import SignDetectorPipeline

async def main():
    print("Initializing pipeline...")
    pipeline = SignDetectorPipeline()
    
    print("\nReading test image (using Data/yes/Image_1779161619.2432585.jpg)...")
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data", "yes", "Image_1779161619.2432585.jpg"), "rb") as image_file:
            b64 = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error reading image: {e}")
        return
        
    print("\nRunning detect_sign (with missing lmList to trigger CNN and LLM fallback)...")
    res = await pipeline.detect_sign(b64, lmList=None)
    
    print("\nResult:", res)
    print("Test Complete.")

if __name__ == "__main__":
    asyncio.run(main())
