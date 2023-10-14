import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

def main():
    st.title("Sign Language Translator")
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    classifier = Classifier("files/keras_model.h5", "files/labels.txt")

    offset = 20
    imgSize = 300
    word_creation_time = None
    current_word = ""
    sentence = []

    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
              'W', 'X', 'Y', 'Z']

    stframe = st.image([])
    sentence_output = st.empty()

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape
            if imgCrop.shape[0] <= 0 or imgCrop.shape[1] <= 0:
                continue

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            if word_creation_time is None:
                word_creation_time = time.time()

            if time.time() - word_creation_time > 3:  # If a sign stays more than 3 seconds
                current_word += labels[index]
                word_creation_time = None  # Reset the word creation timer

        else:
            if word_creation_time is not None:  # If no hand sign detected
                if current_word:  # If there is a partially formed word
                    sentence.append(current_word)
                    current_word = ""
                else:  # If no sign is displayed, write a blank
                    sentence.append('')
                word_creation_time = None  # Reset the word creation timer

        # Display the current word and sentence
        cv2.putText(imgOutput, current_word, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        sentence_output.text(" ".join(sentence))

        # Convert the image for display in Streamlit
        stframe.image(imgOutput, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    final_sentence = " ".join(sentence)
    st.success("Final Sentence: " + final_sentence)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
