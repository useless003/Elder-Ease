import streamlit as st
import speech_recognition as sr


def main():
    st.title("Speech to Text Conversion")

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    lang = st.selectbox("Choose Your Language",("te","hi","ta","en-in"))

    with st.container():
        st.subheader("Instructions")
        st.write("1. Click the 'Start Recording' button to start recording.")
        st.write("2. Speak into your microphone while the recording is active.")
        st.write("3. Click the 'Stop Recording' button to stop recording.")

    if st.button("Start Recording"):
        with microphone as source:
            st.write("Recording...")
            audio = recognizer.listen(source)
        st.write("Recording stopped.")

        st.subheader("Converted Text")
        try:
            text = recognizer.recognize_google(audio,language=lang)
            st.markdown(text)
        except sr.UnknownValueError:
            st.write("Could not understand audio.")
        except sr.RequestError as e:
            st.write(f"Error requesting results from Google Speech Recognition service: {e}")


if __name__ == "__main__":
    main()
