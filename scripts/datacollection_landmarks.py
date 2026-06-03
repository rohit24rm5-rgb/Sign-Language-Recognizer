import sys
import os
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import json
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "DataLandmarks")
if not os.path.exists(folder):
    os.makedirs(folder)

def normalize_landmarks(lmList):
    if not lmList: return None
    try:
        base_x, base_y = lmList[0][0], lmList[0][1]
        if len(lmList[0]) > 2:
            base_x, base_y = lmList[0][1], lmList[0][2]
            return [[lm[1]-base_x, lm[2]-base_y] for lm in lmList]
        return [[lm[0]-base_x, lm[1]-base_y] for lm in lmList]
    except:
        return lmList

# List of signs to collect
print("====================================")
print("Welcome to Landmark Data Collection!")
print("====================================")
print("Enter the signs you want to collect, separated by commas.")
print("Example: Hello, ThumbsUp, Peace, A, B, C")
signs_input = input("> ")
signs = [s.strip() for s in signs_input.split(",") if s.strip()]

if not signs:
    print("No signs entered. Exiting...")
    exit()

print(f"\nSigns to collect: {', '.join(signs)}")
print("Press 's' to save the current hand pose.")
print("Press 'n' to move to the next sign.")
print("Press 'q' to quit.")

current_sign_index = 0
counter = 0

data_store = []

while True:
    current_sign = signs[current_sign_index]
    success, img = cap.read()
    
    # Draw instructions
    cv2.putText(img, f"Collecting: {current_sign}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, f"Count: {counter}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    hands, img = detector.findHands(img, draw=True)

    cv2.imshow("Data Collection", img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        if hands:
            hand = hands[0]
            lmList = hand['lmList']
            vec = normalize_landmarks(lmList)
            if vec:
                # Flatten the vector
                flattened = np.array(vec).flatten().tolist()
                data_store.append({
                    "label": current_sign,
                    "landmarks": flattened
                })
                counter += 1
                print(f"Saved {counter} for {current_sign}")

    elif key == ord("n"):
        current_sign_index += 1
        counter = 0
        if current_sign_index >= len(signs):
            print("Finished all signs!")
            break

    elif key == ord("q") or key == 27:
        break

# Save all data to a single JSON file
if data_store:
    out_file = os.path.join(folder, "landmarks_dataset.json")
    with open(out_file, 'w') as f:
        json.dump(data_store, f)
    print(f"Successfully saved {len(data_store)} total landmarks to {out_file}!")

cap.release()
cv2.destroyAllWindows()
