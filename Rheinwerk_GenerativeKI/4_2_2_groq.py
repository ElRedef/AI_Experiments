
#%% packages
print("Hello Groq\n")
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import sys

# %%
# Groq API Key from .env file

def get_api_key():
    load_dotenv('4_2_2_groq.env')
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("Fehler: GROQ_API_KEY nicht in 4_2_2_groq.env gefunden.")
        sys.exit(1)
    return api_key

# %%

MODEL_NAME = 'llama-3.3-70b-versatile'

def main():
    
    api_key = get_api_key()
    model = ChatGroq(model_name=MODEL_NAME,
        temperature=0.5, # controls creativity
        api_key=api_key)

    frage = "Was ist Huggingface? Bitte antworte auf Deutsch."
    try:
        res = model.invoke(frage)
        print(res)
    except Exception as e:
        print(f"Fehler bei der Anfrage: {e}")

if __name__ == "__main__":
    main()



# %%
