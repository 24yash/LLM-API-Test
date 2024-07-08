import streamlit as st
import pandas as pd

st.title("Upload your CSV files")
train_file = st.file_uploader("Choose a train CSV file", type=['csv'])
test_file = st.file_uploader("Choose a test CSV file", type=['csv'])

# Check if both files have been uploaded
if train_file is not None and test_file is not None:
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

  st.write("##### Numerical Features Information")
  st.dataframe(num_info_df)

  st.write("##### Categorical Features Information")
  st.dataframe(cat_info_df)

  st.title("QuickML Bot")

  # Initialize chat history
  if "messages" not in st.session_state:
      st.session_state.messages = []


  # Display chat messages from history on app rerun
  for message in st.session_state.messages:
      with st.chat_message(message["role"]):
          st.markdown(message["content"])


  import google.generativeai as genai

  # to read api key from env
  # import os
  # from dotenv import load_dotenv

  # load_dotenv()

  # google_api_key = os.getenv('GOOGLE_API_KEY')

  # genai.configure(api_key=google_api_key)

  from google.api_core.exceptions import InvalidArgument


  user_api_key = st.text_input("Enter your Google API Key:", type="password")

  if user_api_key:
    try:
      genai.configure(api_key=user_api_key)
      model = genai.GenerativeModel('gemini-1.5-flash-latest')
      # st.success("API Key accepted and model configured!")

      # Initialize the model
      gemini_chat = model.start_chat()
      response1 = gemini_chat.send_message("System Prompt: You are a bot for QuickML Platform. Your job is to communicate with the user and take params from user to call the APIs. Further instructions will be provided in the next message.")
      print('working')
    except InvalidArgument as e:
      print(e)
      st.error("Invalid API Key. Please pass a valid API key.")
      st.stop()
    except Exception as e:
      print(e)
      st.error(f"Error: {e}")
      st.stop()

    response2 = gemini_chat.send_message(
        """
    Instructions
    - Engage in a conversation style without mentioning APIs.
    - Collect user preferences through the conversation.
    - Call the API with all necessary parameters once preferences are collected.
    - Respond to the user indicating the completion of processing without mentioning the API.

    API Call Format
    Response format: "ReplyToUser: Processing is done. ApiCall: <api_name> <params (json)>."

    User Interaction Guidelines
    - Ask for user preferences one by one.
    - Do not repeat questions or mention handling missing values repeatedly.
    - Move on once preferences are given.

    API Parameters
    - All parameters are required.
    - Gather them without overwhelming the user.
    """

        + """ API Spec sheet: openapi: 3.0.3
    info:
      title: QuickML 3.0
      description: |-
        This is a sample QuickML Server based on the OpenAPI 3.0 specification.
      version: 1.0.11
    servers:
      - url: http://127.0.0.1:5000
    paths:
      /process_data:
        post:
          summary: Process data with specified preprocessing options
          requestBody:
            required: true
            content:
              application/json:
                schema:
                  type: object
                  properties:
                    null_handling:
                      type: string
                      description: How to handle null values for numerical columns
                      enum: [drop, mean, median, constant]
                    null_constant:
                      type: number
                      description: Constant value to replace nulls if null_handling is 'constant'
                    null_handling_categorical:
                      type: string
                      description: How to handle null values for categorical columns
                      enum: [drop, mode, constant]
                    null_categorical_constant:
                      type: string
                      description: Constant value to replace nulls in categorical columns if null_handling_categorical is 'constant'
                    scaling:
                      type: string
                      description: Type of scaling to apply
                      enum: [standard, minmax]
                    categorical_handling:
                      type: string
                      description: How to handle categorical variables
                      enum: [onehot, label]
                    target:
                      type: string
                      description: Name of the target column
          responses:
            200:
              description: Data processed successfully
              content:
                application/json:
                  schema:
                    type: object
                    properties:
                      message:
                        type: string
                      filename:
                        type: string
                      target:
                        type: string
                      test_file_path:
                        type: string
      /model_selection:
        post:
          summary: Select and train a model based on the provided type
          requestBody:
            required: true
            content:
              application/json:
                schema:
                  type: object
                  properties:
                    model:
                      type: string
                      description: Type of model to train
                      enum: [linear, random_forest, knn, logistic]
                    target:
                      type: string
                      description: Name of the target column
          responses:
            200:
              description: Model trained successfully
              content:
                application/json:
                  schema:
                    type: object
                    properties:
                      message:
                        type: string
                      model_file:
                        type: string
      /model_trained:
        get:
          summary: Get predictions from the trained model
          responses:
            200:
              description: Model predictions generated successfully
              content:
                application/json:
                  schema:
                    type: object
                    properties:
                      message:
                        type: string
                      predictions_file:
                        type: string """)
    print(response2.text)
    # st.session_state.messages.append({"role": "assistant", "content": response2.text})


    if prompt := st.chat_input("Please tell how would you like the data to be processed?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})


        r = gemini_chat.send_message(prompt)
        # print(gemini_chat.history)
        
        # response = f"QuickML: {r}"
        response = r.text

        # check if the response is an API call
        # handle api call

        # response = f"Agent: {r}"
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

  else:
    st.warning("Please enter your Google API Key.")