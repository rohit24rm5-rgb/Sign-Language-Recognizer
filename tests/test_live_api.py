import sys
import os
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import requests
import threading
import time
import json

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/detect"

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Global state
latest_prediction = "Waiting..."
latest_confidence = 0.0
is_processing = False

def call_api(image_bytes, lm_list_str):
    global latest_prediction, latest_confidence, is_processing
    try:
        files = {"file": ("frame.jpg", image_bytes, "image/jpeg")}
        data = {"landmarks": lm_list_str}
        response = requests.post(API_URL, files=files, data=data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            latest_prediction = result.get('final_sign', 'Unknown')
            latest_confidence = result.get('confidence', 0.0)
            print(f"API Response: {latest_prediction} | Info: {result.get('critic_feedback', '')}")
    except Exception as e:
        print(f"API Error: {e}")
    finally:
        is_processing = False

print("Starting live feed... Press 'q' to quit.")

hand_entry_time = None

while True:
    success, img = cap.read()
    if not success:
        continue
        
    imgOutput = img.copy()
    hands, img = detector.findHands(img, draw=False)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        lmList = hand['lmList']

        if hand_entry_time is None:
            hand_entry_time = time.time()
            latest_prediction = "Stabilizing (1s)..."

        time_held = time.time() - hand_entry_time

        # Draw the UI bounding box
        cv2.rectangle(imgOutput, (x - 20, y - 90), (x - 20 + 350, y - 20), (0, 255, 0), cv2.FILLED)
        
        # After 1 second of holding the sign, fetch the answer (if not already processing)
        if time_held >= 1.0:
            if not is_processing:
                is_processing = True
                
                # We send the full frame as visual context (optional, but good for logs/UI)
                _, buffer = cv2.imencode('.jpg', imgOutput)
                image_bytes = buffer.tobytes()
                lm_list_str = json.dumps(lmList)
                
                threading.Thread(target=call_api, args=(image_bytes, lm_list_str), daemon=True).start()
                
            # Draw actual prediction
            label = f"{latest_prediction} ({latest_confidence:.2f})"
            cv2.putText(imgOutput, label, (x - 10, y - 40), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 2)
        else:
            # Draw stabilizing text
            cv2.putText(imgOutput, f"Stabilizing... {1.0 - time_held:.1f}s", (x - 10, y - 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)

        cv2.rectangle(imgOutput, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)
    else:
        # Reset when hand leaves frame
        hand_entry_time = None
        latest_prediction = "Waiting..."

    cv2.imshow('Live Camera (AI Tester)', imgOutput)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
