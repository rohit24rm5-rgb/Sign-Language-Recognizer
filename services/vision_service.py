import cv2
import base64
import numpy as np
import math
import sys
import os
from cvzone.HandTrackingModule import HandDetector

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ai_pipeline import SignDetectorPipeline

class VisionService:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=1)
        self.pipeline = SignDetectorPipeline()
        self.offset = 20
        self.img_size = 300
        self.last_detected_sign = "Waiting..."
        self.last_confidence = 0.0

    def get_base64_from_image(self, img):
        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode('utf-8')

    def run(self):
        print("Starting Vision Service... Press 'q' to quit.")
        
        while True:
            success, img = self.cap.read()
            if not success:
                break
                
            imgOutput = img.copy()
            hands, img = self.detector.findHands(img, draw=False)
            
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                
                imgWhite = np.ones((self.img_size, self.img_size, 3), np.uint8) * 255
                
                y1 = max(0, y - self.offset)
                y2 = min(img.shape[0], y + h + self.offset)
                x1 = max(0, x - self.offset)
                x2 = min(img.shape[1], x + w + self.offset)
                
                imgCrop = img[y1:y2, x1:x2]
                
                if imgCrop.size != 0:
                    aspectRatio = h / w
                    if aspectRatio > 1:
                        k = self.img_size / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, self.img_size))
                        wGap = math.ceil((self.img_size - wCal) / 2)
                        
                        # Safe assignment to prevent out-of-bounds ValueError
                        w_end = min(self.img_size, wGap + imgResize.shape[1])
                        imgWhite[:, wGap:w_end] = imgResize[:, :w_end - wGap]
                    else:
                        k = self.img_size / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (self.img_size, hCal))
                        hGap = math.ceil((self.img_size - hCal) / 2)
                        
                        # Safe assignment to prevent out-of-bounds ValueError
                        h_end = min(self.img_size, hGap + imgResize.shape[0])
                        imgWhite[hGap:h_end, :] = imgResize[:h_end - hGap, :]
                    
                    # Instead of running every single frame (which would cost too much and exceed rate limits),
                    # we will only trigger the AI Pipeline when the user presses 's'.
                    # Or we could do it every N frames. Let's do it on key press 's' for now to be safe.
                    
                    cv2.rectangle(imgOutput, (x - self.offset, y - self.offset), (x + w + self.offset, y + h + self.offset), (0, 255, 0), 4)
                    
                    # Display the last detected sign
                    cv2.rectangle(imgOutput, (x - self.offset, y - self.offset - 50), (x - self.offset + 300, y - self.offset), (0, 255, 0), cv2.FILLED)
                    cv2.putText(imgOutput, f"{self.last_detected_sign} ({self.last_confidence:.2f})", (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                    
            cv2.putText(imgOutput, "Press 'ESC' to scan via Groq AI", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(imgOutput, "Or press 1-5 to force-save Geometric Template", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
            cv2.imshow("Image", imgOutput)
            
            key = cv2.waitKey(1)
            
            if key == ord('q'):
                break
                
            # Manual Calibration Hotkeys
            elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')] and hands and imgCrop.size != 0:
                mapping = {ord('1'): "Hello", ord('2'): "iloveyou", ord('3'): "yes", ord('4'): "No", ord('5'): "Thankyou"}
                sign_name = mapping[key]
                print(f"Manual Calibration: Force-saving template for {sign_name}...")
                lmList = hands[0]['lmList']
                vec = self.pipeline.normalize_landmarks(lmList)
                if vec:
                    self.pipeline.memory.save_template(sign_name, vec)
                    self.last_detected_sign = sign_name
                    self.last_confidence = 1.0
                    print(f"Successfully saved {sign_name} geometry to memory!")
                    
            elif key == 27 and hands and imgCrop.size != 0:
                print("Detecting...")
                base64_image = self.get_base64_from_image(imgWhite)
                try:
                    import asyncio
                    result = asyncio.run(self.pipeline.detect_sign(base64_image, hand['lmList']))
                    self.last_detected_sign = result['final_sign']
                    self.last_confidence = result['confidence']
                    print(f"Detected: {self.last_detected_sign} (Confidence: {self.last_confidence})")
                    print(f"Critic Feedback: {result['critic_feedback']}")
                except Exception as e:
                    self.last_detected_sign = "Error"
                    print(f"Error calling LLM: {e}")

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    service = VisionService()
    service.run()
