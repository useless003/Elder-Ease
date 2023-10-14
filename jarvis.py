import pyttsx3
import speech_recognition as sr
import datetime
import os
import wikipedia
import pywhatkit
import pyautogui
import time
import subprocess
import pyjokes
import openai
from config import apikey
import random
from googletrans import Translator
from text_summarization import text_summarizer
import webbrowser
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pyautogui
import time
import autopy

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voices',voices[1].id)
name = "JARVIS"
my_name = "KUMAARA SWAMY"





def Speak(audio):
    engine.say(audio)
    engine.runAndWait()


def fetch_wikipedia_summary(query):
    wikipedia.set_user_agent("CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)")
    try:
        page = wikipedia.page(query)
        summary = page.summary
        # Split the summary based on periods and take the first few sentences
        sentences = summary.split('.')
        short_summary = '.'.join(sentences[:3])  # Use the first 3 sentences
        return short_summary
    except wikipedia.exceptions.DisambiguationError as e:
        return "There are multiple possible results. Please specify your query."
    except wikipedia.exceptions.PageError as e:
        return "The page does not exist. Please check your query."
    except Exception as e:
        return "An error occurred while searching. Please try again later."




def commands():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source,duration=1)
        audio = r.listen(source)
    try:
        print("Wait for few Moments..")
        query = r.recognize_google(audio,language='en-in')
        print(f"You said : {query}\n")
    except Exception as e:
        print(e)
        Speak("Please Tell me again")
        query = 'none'

    return query



def Listen():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening.... ")
        r.pause_threshold = 1
        audio = r.listen(source,0,10)
    try:
        print("Recognizing...")
        query = r.recognize_google(audio,language="te")
    except:
        return ""
    query = str(query).lower()
    return query

#print(Listen())

#________________Translation________________
def TranslationTel2Eng(Text):
    line = str(Text)
    try:
        translate = Translator()
        result = translate.translate(line, "en")
        data = result.text
        return data
    except Exception as e:
        print("Translation Error:", e)
        Speak("Translation Error: Unable to translate the text.")


#doubt = Listen()
#TranslationTel2Eng("నమస్తే")

#________Connect____

openai.api_key = apikey
messages = [
  {"role": "system","content" : "You are a kind helpful assistant"}
]

def chat(prompt):

      #print("Hi There ! I am Chat GPT AI, How Can I Assist you?")
      message = prompt
      if message:
        messages.append(
          {"role":"user","content":message},
        )
        chat = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",messages=messages
        )
        reply = chat.choices[0].message.content
        print(reply)
        Speak(reply)

        messages.append({"role":"assistant","content":reply})

def open_notepad_and_type(query):
    try:
        subprocess.Popen('notepad.exe')  # Open Notepad using subprocess
        time.sleep(2)  # Wait for 2 seconds to allow Notepad to open
        pyautogui.write(query)  # Type the content into Notepad
        return True
    except Exception as e:
        print("Error:", e)
        return False

def wishings():
    hour = int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        print("Good Morning BOSS")
        Speak("Good Morning BOSS")
    elif hour>=12 and hour<17:
        print("Good Afternoon BOSS")
        Speak("Good Afternoon BOSS")
    elif hour>=17 and hour<21:
        print("Good Evening BOSS")
        Speak("Good Evening BOSS")
    else:
        print("Good Night BOSS")
        Speak("Good Night BOSS")

def paste_text():
    print("Paste the text you want to summarize and press Enter:")
    text = input()
    return text

def play_song(search_query):
    # Search for tracks
    results = sp.search(q=search_query, type='track', limit=1)

    # Check if search results contain any tracks
    if len(results['tracks']['items']) > 0:
        track = results['tracks']['items'][0]
        track_name = track['name']
        artist_name = track['artists'][0]['name']

        print(f"Playing '{track_name}' by {artist_name}")

        # Get the track URL and open it in a web browser
        track_url = track['external_urls']['spotify']
        webbrowser.open(track_url)
    else:
        print(f"No results found for '{search_query}'")


if __name__ == "__main__":
    #wishings()

    while True:
        query = commands().lower()
        if "wake up" in query:
            wishings()
            Speak("Yes BOSS what can I do for you?")
            while True:
                query = commands().lower()
                if "time" in query:
                    strTime = datetime.datetime.now().strftime("%H:%M")
                    Speak(f"Sir the time is {strTime}")
                    print(strTime)
                elif "open downloads" in query.lower():
                    Speak("Opening Downloads Sir")
                    os.startfile("C:\\Users\\kumar\\Downloads")

                elif "play movie" in query.lower() or  "play a movie" in query.lower():
                    Speak("Sure Sir playing Samajavaragamana movie")
                    os.startfile("C:\\Users\\kumar\\Videos\\Samajavaragamana (2023) 1080p AHA WEB-DL x265 -Telugu (DD- 5.1 - 384Kbps & AAC 2.0)- Esub_Hamilton_.mkv")

                elif 'open chrome' in query.lower():
                    Speak("Opening Google Chrome Sir!")
                    os.startfile("C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs\\Google Chrome.lnk")
                    while True:
                        chromeQuery = commands().lower()
                        if "search" in chromeQuery:
                            youtubeQuery = chromeQuery
                            youtubeQuery=youtubeQuery.replace("search","")
                            pyautogui.write(youtubeQuery)
                            pyautogui.press('enter')
                            Speak("Searching")
                        elif "close chrome" in chromeQuery or "exit chrome" in chromeQuery or "exit google" in chromeQuery:
                            pyautogui.hotkey("ctrl","w")
                            Speak("Closing Google Chrome Sir! ")
                            break

         ###############################################################################
                elif "ai".lower() in query.lower():
                    print("Hi There ! I am Chat GPT A.I ,I am here to help you")
                    Speak("Hi There ! I am Chat GPT AI, How Can I Assist you?")
                    print("Note : You can talk to me in your regional language and I will respond in English")
                    Speak("Note : You can talk to me in your regional language and I will respond in English")

                    while True:
                        text = Listen()
                        prompt = TranslationTel2Eng(text)
                        print(f"You asked : {prompt}")
                        chat(prompt)  # Call chat() function with the user's input prompt
                        if "exit chat gpt".lower() in prompt.lower() or "close chat gpt".lower() in prompt.lower():
                            print("Exiting Chat GPT....")
                            Speak("Exiting Chat GPT....")
                            break

            #########################################################################
                elif "thank you" in query.lower() or "nice" in query.lower() or "cool" in query.lower():
                    Speak("Yes Sir ! That's my pleasure")



                elif 'open spotify' in query:
                    try:
                        subprocess.Popen(['spotify'])
                        Speak("Spotify opened successfully.")
                    except Exception as e:
                        Speak("Error:", e)
                elif "play song" in query.lower() or "play on spotify" in query.lower() or "play a song" in query.lower():
                    Speak("Yes Sir...Which song would you like me to play?")
                    song = commands()
                    Speak("Playing "+song)
                    play_song(song)
                    time.sleep(8)
                    pyautogui.click(835,1069)
                    while True:
                        command = commands()
                        if "exit song" in command.lower() or "close song" in command.lower():
                            Speak("Exiting from the song...")
                            pyautogui.hotkey("ctrl","w")
                            break
                        #elif "pause" in command.lower() or "pause song" in command.lower():
                            #Speak("Pausing the Song....")
                            #pyautogui.click(1610,1898)
                        #elif "next" in command.lower() or "next song" in command.lower():
                            #Speak("Playing next song...")
                            #pyautogui.click(1707,1871)

                elif 'open pycharm' in query:
                    try:
                        subprocess.Popen('pycharm')
                        print("PyCharm opened successfully.")
                    except Exception as e:
                        print("Error:", e)
                    while True:
                        command = commands()
                        if "exit pycharm" in command.lower() or "close pycharm" in command.lower():
                            Speak("Exiting PyCharm")
                            pyautogui.hotkey("alt","F4")
                            break

                elif "open files" in query.lower():
                    try:
                        os.system("explorer")  # Open File Explorer using os.system
                        Speak("File Explorer Opened")
                    except Exception as e:
                        Speak("Error: " + str(e))
                    while True:
                        command = commands()
                        if "exit files" in command.lower() or "close files" in command.lower():
                            Speak('Ok Sir. Exiting File Explorer')
                            pyautogui.hotkey("alt","F4")
                            break


                elif "open settings" in query.lower():
                    try:
                        if os.name == 'nt':  # Check if the OS is Windows
                            os.system('start ms-settings:')
                            Speak("Settings Opened Successfully")
                        else:
                            Speak("Sorry, opening settings is supported only on Windows.")
                    except Exception as e:
                        Speak("Error: " + str(e))
                    while True:
                        command = commands()
                        if "exit settings" in command.lower() or "close settings" in command.lower():
                            Speak("Exiting Settings")
                            pyautogui.hotkey("alt","F4")
                            break



                elif "mute" in query:
                    Speak("Ok Sir I am Muting")
                    break

                elif 'wikipedia' in query:
                    Speak("Searching in Wikipedia...")
                    try:
                        query = query.replace("wikipedia","")
                        results = fetch_wikipedia_summary(query)
                        if results:
                            Speak("According to Wikipedia,")
                            print(results)
                            Speak(results)
                        else:
                            Speak("No results Found.")
                            print("No results found")
                    except Exception as e:
                        Speak("An error occurred while searching. Please try again later.")
                        print("Error:", e)

                elif 'play on youtube' in query:
                    Speak("What would you like me to play on YouTube?")
                    youtube = commands()
                    Speak("Playing "+youtube)
                    pywhatkit.playonyt(youtube)
                    if "close youtube" in query.lower() or "exit youtube" in query.lower():
                        pyautogui.hotkey("ctrl","w")
                        break

                elif "maximize" in query or "maximise" in query:
                    Speak("Maximizing Sir")
                    pyautogui.hotkey("win","up","up")

                elif "minimize" in query or "minimise" in query:
                    Speak("Minimizing Sir...")
                    pyautogui.hotkey("win","down","down")

                elif "screenshot" in query:
                    Speak("Taking a screenshot Sir!")
                    pyautogui.hotkey("win","prtsc")

                elif "type" in query:
                    query = query.replace("type", "")
                    Speak("Please tell me What should I type?")
                    while True:
                        WriteinNotepad = commands()
                        if WriteinNotepad.lower() == "exit typing":
                            Speak("Ok Sir!")
                            break  # Exit the while loop if "exit typing" is spoken
                        elif "save the file" in query.lower() or "save file" in query.lower() or "save this file" in query.lower():
                            pyautogui.hotkey("ctrl","s")
                            Speak("Sir , Please Specify a name for this file")
                            notepadsavings = commands()
                            pyautogui.write(notepadsavings)
                            pyautogui.press("enter")
                        elif "exit notepad" in query.lower() or "close notepad" in query.lower():
                            Speak("Exiting NotePad..")
                            pyautogui.hotkey("alt","F4")
                            break
                        if open_notepad_and_type(WriteinNotepad):
                            Speak("Successfully typed into Notepad.")
                        else:
                            Speak("Failed to type into Notepad.")

                elif 'joke' in query:
                    joke = pyjokes.get_joke()
                    print(joke)
                    Speak(joke)
                elif "your name" in query.lower():
                    Speak(f"My name is {name} and I am created by {my_name}")
                elif "how are you" in query.lower():
                    Speak("I am Fine. I am Glad that you have asked")
                elif "change your name" in query.lower():
                    Speak("Of Course! What name would you like to give me?")
                    new_name = commands()
                    if new_name:
                        Speak(f"My name have been changed to {new_name}")
                    else:
                        Speak(f"You did not give me a new name. So my name will be {name}")
                    name = new_name()
                elif "summarise" in query.lower() or "summarize" in query.lower() or "summarization" in query.lower() or "summarisation" in query.lower():
                    Speak("Yes Sir Ofcourse! Please give me the necessary text to summarize")
                    text = paste_text()
                    summarization = text_summarizer(text)
                    print("\n")
                    print(summarization)
                    Speak(summarization)
                elif "open" in query.lower():
                    query = query.replace("open","")
                    pyautogui.press("win")
                    pyautogui.write(query)
                    pyautogui.press("enter")
                    Speak(f"Successfully Opened {query}")
                elif 'exit program' in query.lower() or "close program" in query.lower() or "exit the program" in query.lower():
                    Speak("Leaving Sir.......")
                    quit()
                elif "exit" in query.lower():
                    query = query.replace("exit","")
                    Speak(f"Exiting from {query}")
                    pyautogui.hotkey("alt","F4")
                elif "hey" in query.lower():
                    Speak("Yes Sir..")



