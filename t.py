import streamlit as st
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

cohere_api_key = os.getenv('COHERE_API_KEY')

llm = ChatCohere(cohere_api_key=cohere_api_key)

import yaml

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world-class machine learning expert. Your job is to communicate with the user and take params from user to call the APIs. The API spec sheet is as follows: . When calling API: response=API_CALL:<> ,REPLY_TO_USER: <> "),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

st.title('Langchain-Cohere Chat')

user_input = st.text_input("You:", "")

if st.button('Send'):
    response = chain.invoke({"input": user_input})
    
    st.text_area("AI:", value=response, height=100, max_chars=None, key=None)