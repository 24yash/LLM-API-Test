import streamlit as st


def train_model(name):
    st.write(f"Training model: {name}")
    return f"Model {name} trained successfully"

from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

cohere_api_key = os.getenv('COHERE_API_KEY')

llm = ChatCohere(cohere_api_key=cohere_api_key)


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a chatbot. Your job is to communicate with the user and get the type of ML Model they want to train. You must not do anything other than this purpose but you can help the user make an informed choice. Call the function when required, which is: 'train_model'. This function takes a parameter of string type that is 'name', which is basically the name of the ML Model which can be one of 4 values: LR, RF, KNN, SVM which stands for Linear Regression, Random Forest, K-Nearest Neighbours and Support vector machine respectively. When calling the function to train your response should only be <function_name(<paramater>)>. You do not have to tell the user directly about the function or about yourself or the app. Just talk to user to get their choice and in response call the function with the parameter. "),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser


st.title("Echo Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    r = chain.invoke({"input": prompt})

    #  add check if r contains call to fn train_model then instead of adding to session state, call the actual fn: train_model with the param  

    if "train_model" in r:
        print('if')
        print(r)
        # Extract the model name from the response
        model_name = r.split("train_model('")[1].split("')")[0]
        # Call train_model with the extracted model name
        train_model_result = train_model(model_name)
        response = f"QuickML: {train_model_result}"
    else:
        print('else')
        print(r)
        response = f"QuickML: {r}"

    # response = f"Agent: {r}"
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})