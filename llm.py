import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

import os
from dotenv import load_dotenv

load_dotenv()

google_api_key = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=google_api_key)

# Loading Gemini Pro
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Initialize the model
gemini_chat = model.start_chat()

# We are going to send the first message
response1 = gemini_chat.send_message("Hello my name is hamza")
print(response1.text)

# We are going to send the second message
response2 = gemini_chat.send_message("What is my name?")
print(response2.text)

