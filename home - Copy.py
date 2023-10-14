import streamlit as st
from streamlit_option_menu import option_menu
import streamlit as st
import streamlit_authenticator as stauth
from dependencies import sign_up,fetch_user
from streamlit_lottie import st_lottie
import requests
from twilio.rest import Client


selected=option_menu(
        menu_title=None,
        options=["Home","About","Login/SignUp"],
        icons=['house','chat-quote','journal-code'],
        default_index=0,
        orientation="horizontal"
    )


def lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def twilio():
    account_sid = "AC01f793ae377e237a2ec273b06e730be7"
    auth_token = "d433ff50ff347f4bd362b9a614d2aae7"
    client = Client(account_sid,auth_token)
    call = client.calls.create(
        twiml = '<Response><Say>Its an emergency.Its an Emergency.Its an Emergency</Say></Response>',
        to = '+919346502352',
        from_ = "+19365876737"
    )
    print(call.sid)


with st.sidebar:
    lottie_animation11 = "https://lottie.host/97a2d6de-b047-4e89-9294-5f1e70ae8550/vjDo3tyMZQ.json"
    lottie_json11 = lottie_url(lottie_animation11)
    st_lottie(lottie_json11,key="logo",height=250,width=250)
    option_menu(
        menu_title=None,
        options=["ElderEase"],
        icons=['alexa']
    )
    st.markdown(":green[-Caring for Seniors, One Click at a Time.🫡]")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")

    if st.sidebar.button("Emergency"):
        twilio()


    #############################HOME##############################
if selected == "Home":
    st.title("👴 Welcome to ElderEase 🧓")





    lottie_animation = "https://lottie.host/1a128bcd-8b39-4798-a776-7d92da3a732b/SLotOXOGpI.json"
    lottie_json = lottie_url(lottie_animation)
    st_lottie(lottie_json, key="welcome")

    st.header("Empowering Seniors for a Better Life")
    st.write("Are you or a loved one looking for support in health and communication? ElderEase is here to help.")
    st.markdown("- 👵 Stay Healthy: Explore our curated health resources, exercise routines, and dietary tips to maintain your well-being.")
    st.markdown("- 🗣️ Stay Connected: Seamlessly connect with friends and family through our user-friendly communication tools.")
    st.markdown("- 🤗 Simple & Intuitive: Our platform is designed with seniors in mind. Easy to use, no tech hassle.")
    st.markdown("- 🌟 Your Trusted Companion: Count on ElderEase for guidance, companionship, and assistance.")


    st.markdown('''
    <style>
    [data-testid="stMarkdownContainer"] ul{
        list-style-position: inside;
    }
    </style>
    ''', unsafe_allow_html=True)
    st.write("Join us in enhancing the golden years of life. Start your journey today!")




#############################ABOUT##############################
if selected=="About":
    st.title(":blue[About ElderEase]")
    lottie_animation2 ="https://lottie.host/fef086f8-1574-4caf-ba7b-fa7c12ced535/gpPTP1xupV.json"
    lottie_json2 = lottie_url(lottie_animation2)
    st_lottie(lottie_json2,key="about us")
    st.header("Empowering Seniors for a Better Life")
    st.markdown("At ElderEase, we believe that every stage of life should be cherished, especially the golden years. Our mission is to empower seniors to live healthier, happier, and more connected lives.")
    st.subheader("Who We Are:")
    st.markdown("ElderEase was founded by a group of passionate individuals who recognized the unique challenges faced by seniors in today's fast-paced world. With backgrounds in healthcare, technology, and senior care, our team is dedicated to making a positive impact on the lives of elders.")
    st.subheader("Our Vision:")
    st.markdown("We envision a world where seniors have easy access to the tools and resources they need to age gracefully and maintain strong connections with their loved ones.")
    st.subheader("What We Offer:")
    st.markdown(
        "-Health and Wellness: Discover a wealth of resources, articles, and exercise routines tailored to senior health. We're here to help you lead an active and healthy lifestyle. ")
    st.markdown(
        "- Communication Made Simple: Our user-friendly communication tools make it effortless to stay in touch with family and friends, no matter where they are.")
    st.markdown("- Tech for All Ages: We're committed to making technology accessible to everyone. Our platform is designed with seniors' needs in mind, ensuring a smooth and frustration-free experience.")

    st.markdown('''
        <style>
        [data-testid="stMarkdownContainer"] ul{
            list-style-position: inside;
        }
        </style>
        ''', unsafe_allow_html=True)
    st.subheader("Join Our Community:")
    st.markdown("ElderEase is more than just a platform; it's a community. Join us today and be part of a growing network of seniors and caregivers who are embracing technology to enhance their lives.")
    st.markdown("Thank you for choosing ElderEase. Together, we're redefining the way seniors experience the world.")
    st.markdown("\n")
    st.markdown("\n")
    b1,b2,b3 = st.columns(3)
    with b3:
        choosen = option_menu(
            menu_title=None,
            options=["Chat With Us"],
            icons=['chat']
        )
    #######################################################################################################################
    if choosen=="Chat With Us":
        import streamlit as st
        from streamlit_chat import message

        from langchain.chat_models import ChatOpenAI
        from langchain.schema import (
            SystemMessage,
            HumanMessage,
            AIMessage
        )
        from googletrans import Translator


        def main():

            chat = ChatOpenAI(openai_api_key="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",
                                model='gpt-3.5-turbo')

            if "messages" not in st.session_state:
                st.session_state.messages = [
                    SystemMessage(content="You are a helpful medical assistant.")

                ]



            st.header("Your Own Health Assistant 🤖")
            lottie_animation3 = "https://lottie.host/4ffc0bb2-103a-41d4-90f1-c44ca18df8e6/PebERUh9SG.json"
            lottie_json3 = lottie_url(lottie_animation3)
            st_lottie(lottie_json3,key="chat with us")

            lang = st.selectbox("Choose Your Language - Message", ("te", "hi", "en-in"), key="lang_")
            user_input = st.text_input("Your message: ", key="user_input")
            if user_input:
                st.session_state.messages.append(HumanMessage(content=user_input))
                with st.spinner():
                    response = chat(st.session_state.messages)
                    st.session_state.messages.append(AIMessage(content=response.content))

            messages = st.session_state.get('messages', [])

            for i, msg in enumerate(messages[1:]):

                if i % 2 == 0:
                    message(msg.content, is_user=True, key=str(i) + '_user')
                else:
                    bot_msg = msg.content
                    line = str(bot_msg)
                    # message(msg.content,is_user=False,key=str(i)+'_ai')
                    if lang == "en-in":
                        message(msg.content, is_user=False)
                    elif lang == "te":
                        try:
                            translate = Translator()
                            result = translate.translate(line, "te")
                            data = result.text
                            message(data, is_user=False)
                        except Exception as e:
                            print("Translation Error:", e)
                            message("Translation Error: Unable to translate the text.", is_user=False,
                                    key=str(i) + "_telugu message")
                    elif lang == "hi":
                        try:
                            translate = Translator()
                            result = translate.translate(line, "hi")
                            data = result.text
                            message(data, is_user=False)
                        except Exception as e:
                            print("Translation Error:", e)
                            message("Translation Error: Unable to translate the text.", is_user=False,
                                    key=str(i) + "_hindi message")


        if __name__ == "__main__":
            main()
##########################################################################################################################





if selected=="Login/SignUp":
    st.title(f"You have selected {selected}")
    st.markdown(":blue[Please Login/SignUp To Use the main Features]")
    try:
        users = fetch_user()
        emails = []
        usernames = []
        passwords = []

        for user in users:
            emails.append(user['key'])
            usernames.append(user['username'])
            passwords.append(user['password'])

        credentials = {'usernames':{}}
        for index in range(len(emails)):
            credentials['usernames'][usernames[index]] = {"name":emails[index],'password':passwords[index]}
        Authenticator = stauth.Authenticate(credentials,cookie_name='Streamlit',key='abcdef',cookie_expiry_days=4)

        email,authentication_status,username = Authenticator.login(":green[Login]","main")

        info ,info1 = st.columns(2)

        if not authentication_status:
            sign_up()

        if username:
            if username in usernames:
                if authentication_status:
                    #let user see app
                    st.title(":green[Communicate]")
                    taken = option_menu(
                        menu_title=None,
                        options = ["ElderEase","Sign Language Communication",'Text to Speech',"Speech to Text","Voice To Voice Bot","Health Guide","Reminder"],
                        icons = ['alexa','person-bounding-box','megaphone-fill','chat-text','file-earmark-text','hospital','bell']
                    )
                    if taken =="ElderEase":
                        st.title("Empowering Elders for a Life of Comfort and Connection with ElderEase.")
                        lottie_animation10 = "https://lottie.host/eaedf12a-a5a4-49af-9e36-3589b539c6b0/nyBVrjsoIr.json"
                        lottie_json10 = lottie_url(lottie_animation10)
                        st_lottie(lottie_json10,key = "ElderEase")
                    ###################################################################################
                    if taken=="Sign Language Communication":
                        st.title("Sign Language Communication")
                        lottie_animation4 = "https://lottie.host/4aa58263-bff9-4f52-8d1d-2457a8b4b043/9uJsNrYZVc.json"
                        lottie_json4 = lottie_url(lottie_animation4)
                        st_lottie(lottie_json4,key="sign",height=500,width=500)
                        st.subheader("There are several reasons why we might choose to focus on American Sign Language (ASL) alphabets in our model for sign language communication:")
                        st.markdown("-Widespread Usage ")
                        st.markdown("-Accessibility ")
                        st.markdown("-Educational Resources ")
                        st.markdown("-International Recognition ")

                        st.markdown('''
                            <style>
                            [data-testid="stMarkdownContainer"] ul{
                                list-style-position: inside;
                            }
                            </style>
                            ''', unsafe_allow_html=True)


                        import cv2
                        from cvzone.HandTrackingModule import HandDetector
                        from cvzone.ClassificationModule import Classifier
                        import numpy as np
                        import math
                        import time


                        def main():
                            #st.title("Sign Language Translator")
                            cap = cv2.VideoCapture(0)
                            detector = HandDetector(maxHands=1)
                            classifier = Classifier("files/keras_model.h5", "files/labels.txt")

                            offset = 20
                            imgSize = 300
                            word_creation_time = None
                            current_word = ""
                            sentence = []

                            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                                      'Q', 'R', 'S', 'T', 'U', 'V',
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
                                cv2.putText(imgOutput, current_word, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),
                                            2)
                                sentence_output.text(" ".join(sentence))

                                # Convert the image for display in Streamlit
                                stframe.image(imgOutput, channels="BGR")

                            final_sentence = " ".join(sentence)
                            st.success("Final Sentence: " + final_sentence,key="success")
                            cap.release()
                            cv2.destroyAllWindows()

                        if __name__ == "__main__":
                            main()

                    #######################################################################################
                    if taken=="Text to Speech":
                        st.title("Text to Speech")
                        lottie_animation5 = "https://lottie.host/19aa8bd4-aa4b-4cdc-bb75-cfcc4e2a74c1/RTrsJjZ7Gu.json"
                        lottie_json5 = lottie_url(lottie_animation5)
                        st_lottie(lottie_json5,key="text",height=400,width=400)
                        from gtts import gTTS
                        import tempfile
                        import os


                        def main():
                            st.title("Text to Speech Conversion")

                            text_input = st.text_area("Enter text to convert to speech:")
                            if st.button("Convert to Speech"):
                                if text_input:
                                    speech = gTTS(text_input)
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                                        temp_file_path = f.name
                                        speech.save(temp_file_path)
                                    st.audio(temp_file_path, format="audio/mp3")
                                    os.remove(temp_file_path)

                        if __name__ == "__main__":
                            main()

                    ########################################################################################
                    if taken == "Speech to Text":
                        st.title("Speech to Text")
                        lottie_animation6 = "https://lottie.host/ccab2abc-8baf-4319-b638-2e08121af3f1/1nzFNegokS.json"
                        lottie_json5=lottie_url(lottie_animation6)
                        st_lottie(lottie_json5,key="convert",height=400,width=400)
                        import speech_recognition as sr

                        def main():
                            st.title("Speech to Text Conversion")

                            recognizer = sr.Recognizer()
                            microphone = sr.Microphone()
                            lang = st.selectbox("Choose Your Language", ("te", "hi", "ta", "en-in"))

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
                                    text = recognizer.recognize_google(audio, language=lang)
                                    st.subheader(text)
                                except sr.UnknownValueError:
                                    st.write("Could not understand audio.")
                                except sr.RequestError as e:
                                    st.write(f"Error requesting results from Google Speech Recognition service: {e}")


                        if __name__ == "__main__":
                            main()
                    ############################################################################################
                    if taken=="Voice To Voice Bot":
                        st.title("Voice To Voice Assistant")
                        lottie_animation9 = "https://lottie.host/ff34edf7-97e3-4d62-b673-56a56e0f7b85/VpVTodgMjT.json"
                        lottie_json9 = lottie_url(lottie_animation9)
                        st_lottie(lottie_json9,key="bot",width =500,height=500)

                        st.subheader(":blue[This bot can talk back to you in your comfortable language]")

                        import streamlit as st
                        from streamlit_chat import message
                        #from apikey import OPENAI_API_KEY
                        import speech_recognition as sr
                        from gtts import gTTS
                        import tempfile
                        import os

                        from langchain.chat_models import ChatOpenAI
                        from langchain.schema import (
                            SystemMessage,
                            HumanMessage,
                            AIMessage
                        )


                        def main():

                            chat = ChatOpenAI(openai_api_key="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",
                                              model='gpt-3.5-turbo')

                            if "messages" not in st.session_state:
                                st.session_state.messages = [
                                    SystemMessage(content="You are a helpful kind assistant for mostly old people.")

                                ]


                            st.header("Voice to Voice Assistant 🤖")

                            recognizer = sr.Recognizer()
                            microphone = sr.Microphone()
                            lang = st.selectbox("Choose Your Language", ("te", "hi", "ta", "en-in"))
                            text = ""

                            if st.button("Start Recording"):
                                with microphone as source:
                                    st.write("Recording...")
                                    audio = recognizer.listen(source)
                                st.write("Recording Stopped")

                                try:
                                    text = recognizer.recognize_google(audio, language=lang)
                                except sr.UnknownValueError:
                                    st.write("Could not understand audio.")
                                except sr.RequestError as e:
                                    st.write(f"Error requesting results from Google Speech Recognition service: {e}")

                            user_input = text
                            if user_input:
                                st.session_state.messages.append(HumanMessage(content=user_input))
                                with st.spinner():
                                    response = chat(st.session_state.messages)
                                st.session_state.messages.append(AIMessage(content=response.content))

                            messages = st.session_state.get('messages', [])
                            for i, msg in enumerate(messages[1:]):
                                if i % 2 == 0:
                                    message(msg.content, is_user=True, key=str(i) + '_user')
                                else:
                                    message(msg.content, is_user=False, key=str(i) + '_ai')
                                    ai_msg = msg.content
                                    if ai_msg:
                                        speech = gTTS(ai_msg)
                                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                                            temp_file_path = f.name
                                            speech.save(temp_file_path)

                                        # Play the saved speech
                                        st.audio(temp_file_path, format="audio/mp3")

                                        # Clean up: remove the temporary file
                                        os.remove(temp_file_path)

                        if __name__ == "__main__":
                            main()
                    #############################################################################################
                    if taken == "Health Guide":
                        st.title("Health Guide")
                        lottie_animation7 = "https://lottie.host/786d9858-2308-427c-888a-0c8162544ac4/QF4NnCVjML.json"
                        lottie_json7 = lottie_url(lottie_animation7)
                        st_lottie(lottie_json7,key="health",height=400,width=500)
                        from langchain.llms import OpenAI
                        from langchain.prompts import PromptTemplate
                        from langchain.chains import LLMChain, SimpleSequentialChain


                        st.title("Health Guide")
                        st.markdown("Here you can get to know to about your health problem by just giving it's name.")
                        prompt = st.text_input("Enter your problem here")

                        title_template = PromptTemplate(
                            input_variables=['topic'],
                            template="Generate a complete reason to cure and cure to prevention steps for {topic}. Keep headings for each Reasons , Cure , Preventions."
                                     "Write Just few points for each stage(reason for {topic} , cure for {topic} and preventions for {topic}."
                                     "In cure give both ayurvedic approach and english medicine approach. "

                        )

                        llm = OpenAI(openai_api_key="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")
                        title_chain = LLMChain(llm=llm, prompt=title_template, verbose=False)

                        if prompt:
                            response = title_chain.run(topic=prompt)
                            st.write(response)
                    ##############################################################################################
                    if taken == "Reminder":
                        st.title("Reminder")
                        lottie_animation8 = "https://lottie.host/b70b1565-ad53-4a21-a17c-d61f669b998b/msc7k1Mmpf.json"
                        lottie_json8 = lottie_url(lottie_animation8)
                        st_lottie(lottie_json8,key="tablet",height=400,width=500)
                        import streamlit as st
                        from datetime import datetime, time, timedelta
                        from winotify import Notification,audio


                        def main():
                            st.title("Windows Notifications App")

                            num_notifications = st.number_input("Enter the number of notifications:", min_value=1,
                                                                value=1, step=1)

                            notifications = []

                            for i in range(num_notifications):
                                st.write(f"Notification {i + 1}")
                                notification_time = st.time_input(f"Time for Notification {i + 1}")
                                message = st.text_input(f"Message for Notification {i + 1}", f"Take Tablet {i + 1}")

                                notifications.append({"time": notification_time, "message": message})

                            if st.button("Set Notifications"):
                                for notification in notifications:
                                    notification_time = notification["time"]
                                    message = notification["message"]

                                    # Calculate time until next notification
                                    current_datetime = datetime.now()
                                    target_datetime = datetime.combine(current_datetime.date(), notification_time)

                                    if target_datetime <= current_datetime:
                                        target_datetime += timedelta(days=1)

                                    time_until_notification = (target_datetime - current_datetime).total_seconds()

                                    st.write(
                                        f"Notification set for {notification_time.strftime('%I:%M %p')} with message: {message}")

                                    # Schedule the notification
                                    toast = Notification(app_id="Notification Alert",
                                                         title="Time to take Tablet",
                                                         msg=message,
                                                         duration="long",
                                                         icon=r"C:\Users\kumar\Downloads\Medicine.jpg")
                                    toast.set_audio(audio.LoopingCall, loop=True)
                                    toast.add_actions(label="Order Online", launch="https://www.netmeds.com/")

                                    toast.show()


                        if __name__ == "__main__":
                            main()

                    st.markdown(
                        '''
                        Created by Team Challengers😁
                        '''
                    )
                    st.sidebar.subheader(f"Welcome {username}")
                    Authenticator.logout('Log Out','sidebar')
                    if Authenticator.logout:
                        st.success("Logged In Successfully")

                elif not authentication_status:
                    with info:
                        st.error("Incorrect Password or Username")
                else:
                    with info:
                        st.warning("Please feed in your Credentials")

            else:
                with info:
                    st.warning("User name does not exist. Please SignUp")


    except:
        st.success("Refresh Page")