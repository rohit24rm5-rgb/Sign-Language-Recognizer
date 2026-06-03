import os
import glob
import cv2
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvzone.HandTrackingModule import HandDetector
from core.memory_manager import MemoryManager
from core.ai_pipeline import SignDetectorPipeline
from core.ai_pipeline import SignDetectorPipeline

def build_templates():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")
    signs = ["Hello", "iloveyou", "yes", "No", "Thankyou"]
    
    detector = HandDetector(maxHands=1)
    pipeline = SignDetectorPipeline()
    memory = MemoryManager()
    
    print("Building perfect geometric templates from Data folder...")
    
    for sign in signs:
        folder = os.path.join(data_dir, sign)
        if not os.path.exists(folder):
            # Sometimes Kaggle dataset might be uppercase or slightly different
            folder = os.path.join(data_dir, sign.upper())
            if not os.path.exists(folder):
                print(f"Warning: Folder for {sign} not found!")
                continue
                
        images = glob.glob(os.path.join(folder, "*.jpg"))
        if not images:
            print(f"Warning: No images found for {sign}")
            continue
            
        vectors = []
        print(f"Processing {sign}... (using up to 50 images)")
        
        # Use up to 50 images to get a robust average
        for img_path in images[:50]:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            hands, _ = detector.findHands(img, draw=False)
            if hands:
                lmList = hands[0]['lmList']
                vec = pipeline.normalize_landmarks(lmList)
                if vec:
                    vectors.append(vec)
                    
        if vectors:
            # Calculate average vector (centroid)
            centroid = np.mean(vectors, axis=0)
            # Re-normalize the centroid
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
                
            memory.save_template(sign, centroid.tolist())
            print(f"Successfully saved perfect template for {sign} based on {len(vectors)} images.")
        else:
            print(f"Failed to find any hands in the images for {sign}.")
            
    print("Template generation complete!")

if __name__ == "__main__":
    build_templates()
