#%%packages
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv('4_2_2_groq.env')
print("OPENAI Test\n")
print(os.getenv('OPENAI_API_KEY'))


# %%
MODEL_NAME = "gpt-4o-mini"
model = ChatOpenAI(model_name=MODEL_NAME, temperature=0, openai_api_key=os.getenv('OPENAI_API_KEY '))
response = model.predict("What is the capital of France?")
print(response)
# %%
