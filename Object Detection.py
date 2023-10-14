from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import streamlit as st

model = YOLO("/Yolo-Weights/yolov8n.pt")

def callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format='bgr24')
    img = cv2.flip(img,1)
    results = model.predict(source=img, show=True)
    return av.VideoFrame.from_ndarray(img,format='bgr24')



webrtc_streamer(key='word',video_frame_callback=callback)