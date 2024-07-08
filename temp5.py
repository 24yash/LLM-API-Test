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

import yaml

with open('openapi.yaml', 'r') as file:
    api_spec = yaml.safe_load(file)

api_spec_str = str(api_spec)
# print(api_spec_str)

ss = (
    "You are a bot for QuickML Platform. Your job is to communicate with the user and take params from user to call the APIs. The API spec sheet is as follows: <api_spec_here>"
    + ". Make this a conversation style with the user where you learn user's preferences and call API only once you know the User's preferences that can be used as the parameter. Don't tell the user about the APIs. When calling API your response should be: ReplyToUser: <add a line to tell that processings is done, dont mention about API.> ApiCall: <api_name> <params body (json) >. Your response should not have any extra information. Just talk to user to get their choice and in response call the API with the parameter."
)


prompt = ChatPromptTemplate.from_messages([
    ("system", ss),
    ("user", "{input}")
])


output_parser = StrOutputParser()

chain = prompt | llm | output_parser

import pandas as pd

st.title("Upload your CSV files")
train_file = st.file_uploader("Choose a train CSV file", type=['csv'])
test_file = st.file_uploader("Choose a test CSV file", type=['csv'])

# Check if both files have been uploaded
if train_file is not None and test_file is not None:
    # Files have been uploaded, you can now read them using pandas or proceed with other operations
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    train_df.to_csv('try/train.csv', index=False)
    test_df.to_csv('try/test.csv', index=False)

    st.write("### Train DataFrame Information")
    
    numerical_features = [column for column in train_df.columns if train_df[column].dtype in ['int64', 'float64']]
    categorical_features = [column for column in train_df.columns if train_df[column].dtype in ['object', 'category']]

    num_info_df = pd.DataFrame({
        "Feature Name": numerical_features,
        "Null Values": train_df[numerical_features].isnull().sum(),
        "Unique Values": train_df[numerical_features].nunique()
    })

    cat_info_df = pd.DataFrame({
        "Feature Name": categorical_features,
        "Null Values": train_df[categorical_features].isnull().sum(),
        "Unique Values": train_df[categorical_features].nunique()
    })

    st.write("#### Numerical Features Information")
    st.dataframe(num_info_df)

    st.write("#### Categorical Features Information")
    st.dataframe(cat_info_df)
st.title("QuickML Bot")

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

    # Create the conversation history to pass to the LLM
    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])

    response = chain.invoke({"input": conversation_history})

    r = response 
    # r = chain.invoke({"input": prompt})

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