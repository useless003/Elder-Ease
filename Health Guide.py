from apikey import OpenAI_Api_Key
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,SimpleSequentialChain
import streamlit as st

st.title("Health Guide")
st.markdown("Here you can get to know to about your health problem by just giving it's name.")
prompt = st.text_input("Enter your problem here")

title_template = PromptTemplate(
    input_variables = ['topic'],
    template = "Generate a complete reason to cure and cure to prevention steps for {topic}. Keep headings for each Reasons , Cure , Preventions."
               "Write Just few points for each stage(reason for {topic} , cure for {topic} and preventions for {topic}."
               "In cure give both ayurvedic approach and english medicine approach. "

)

llm = OpenAI(openai_api_key = "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")
title_chain = LLMChain(llm=llm,prompt=title_template,verbose=False)

if prompt:
    response = title_chain.run(topic = prompt)
    st.write(response)