from transformers import pipeline
import requests
import os
from apikey import *



def img2text(img):
    image_to_text = pipeline("image-to-text",model="Salesforce/blip-image-captioning-base",
                             max_new_tokens = 100)
    text = image_to_text(img)

    #print(text[0]["generated_text"])
    return text[0]["generated_text"]