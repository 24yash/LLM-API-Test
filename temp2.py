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
    ("system", "You are a chatbot. Your job is to communicate with the user and get the type of ML Model they want to train. You must not do anything other than this purpose but you can help the user make an informed choice. Call the function when required, which is: 'train_model'. This function takes a parameter of string type that is 'name', which is basically the name of the ML Model which can be one of 4 values: LR, RF, KNN, SVM which stands for Linear Regression, Random Forest, K-Nearest Neighbours and Support vector machine respectively. You do not have to tell the user directly about the function or about yourself or the app. Just talk to user to get their choice and in response call the function with the parameter. "),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

import pandas as pd

# Create file uploaders for test and train CSV files
st.title("Upload your CSV files")
train_file = st.file_uploader("Choose a train CSV file", type=['csv'])
test_file = st.file_uploader("Choose a test CSV file", type=['csv'])

# Check if both files have been uploaded
if train_file is not None and test_file is not None:
    # Files have been uploaded, you can now read them using pandas or proceed with other operations
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)

    st.write("### Train DataFrame Information")
    
    # Creating a summary DataFrame
    info_df = pd.DataFrame({
        "Feature Name": train_df.columns,
        "Null Values": train_df.isnull().sum(),
        "Unique Values": train_df.nunique()
    })
    
    st.dataframe(info_df)

    # Asking for a target variable as input
    target_variable = st.text_input("Enter the target variable:")
    
    print(target_variable)

    # Storing the input in a Python variable
    if target_variable:
        st.session_state['target_variable'] = target_variable
        st.write(f"Target Variable set to: {target_variable}")
    
    

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
else:
    st.write("Please upload both train and test CSV files to proceed.")