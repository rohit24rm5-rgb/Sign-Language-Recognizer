import sys
import os
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
counter = 0

folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data", "Hello")

while True:
    success, img = cap.read()

    if not success:
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)

                imgResize = cv2.resize(imgCrop, (wCal, imgSize))

                wGap = math.ceil((imgSize - wCal) / 2)

                imgWhite[:, wGap:wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)

                imgResize = cv2.resize(imgCrop, (imgSize, hCal))

                hGap = math.ceil((imgSize - hCal) / 2)

                imgWhite[hGap:hCal + hGap, :] = imgResize

            imgCrop=cv2.resize(imgCrop,(224,224))
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite.copy())

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print("Saved:", counter)

    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()