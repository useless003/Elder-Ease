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

    chat = ChatOpenAI(openai_api_key = "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",
                      model='gpt-3.5-turbo')

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content = "You are a helpful medical assistant.")

        ]

    st.set_page_config(
        page_title="Your Own ChatGPT",
        page_icon="🤖"
    )
    st.header("Your Own Health Assistant 🤖")

    lang = st.selectbox("Choose Your Language - Message", ("te", "hi", "en-in"), key="lang_")
    user_input = st.text_input("Your message: ",key="user_input")
    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner():
            response = chat(st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=response.content))

    messages = st.session_state.get('messages',[])

    for i,msg in enumerate(messages[1:]):

        if i % 2 == 0:
            message(msg.content,is_user=True,key=str(i)+'_user')
        else:
            bot_msg = msg.content
            line = str(bot_msg)
            #message(msg.content,is_user=False,key=str(i)+'_ai')
            if lang=="en-in":
                message(msg.content,is_user=False)
            elif lang=="te":
                try:
                    translate = Translator()
                    result = translate.translate(line, "te")
                    data = result.text
                    message(data,is_user=False)
                except Exception as e:
                    print("Translation Error:", e)
                    message("Translation Error: Unable to translate the text.",is_user=False,key=str(i)+"_telugu message")
            elif lang=="hi":
                try:
                    translate = Translator()
                    result = translate.translate(line, "hi")
                    data = result.text
                    message(data,is_user=False)
                except Exception as e:
                    print("Translation Error:", e)
                    message("Translation Error: Unable to translate the text.",is_user=False,key =str(i)+ "_hindi message")



if __name__ == "__main__":
    main()