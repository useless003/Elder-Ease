import streamlit as st
from gtts import gTTS
import tempfile
import os

def main():
    st.title("Text to Speech Conversion")

    text_input = st.text_area("Enter text to convert to speech:")
    if st.button("Convert to Speech"):
        if text_input:
            # Generate speech
            speech = gTTS(text_input)

            # Save speech to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                temp_file_path = f.name
                speech.save(temp_file_path)

            # Play the saved speech
            st.audio(temp_file_path, format="audio/mp3")

            # Clean up: remove the temporary file
            os.remove(temp_file_path)

if __name__ == "__main__":
    main()
