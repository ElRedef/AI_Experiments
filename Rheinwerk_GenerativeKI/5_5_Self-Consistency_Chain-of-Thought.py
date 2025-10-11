#%% packages
import sys
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
from pprint import pprint
import os

# %%
# %%
# Groq API Key from .env file

def get_api_key():
    load_dotenv('4_2_2_groq.env')
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("Fehler: GROQ_API_KEY nicht in 4_2_2_groq.env gefunden.")
        sys.exit(1)
    return api_key

def chain_of_thought_prompting(prompt: str, model_name: str) -> str:
    model = ChatGroq(model_name = model_name)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Du bist ein hilfreicher Assistent, der präzise Antworten auf komplexe Fragen gibt."),
        ("user", f"{prompt} \n Denke Schritt für Schritt, aber erkläre die Schritte nicht. Gib nur eine einzige Gleichung als Antwort zurück")
    ])
    chain = prompt | model
    return chain.invoke({}).content

def self_consistency_cot(prompt: str, model_name: str, n: int = 5) -> str:
    res = []
    for _ in range(n):
        answer = chain_of_thought_prompting(prompt, model_name)
        print(str(n) + ": " + answer+ "\n")
        res.append(answer)

    res_concat = ";".join(res)
    self_consistency_prompt = f"You will get multiple answes in <<>>, separated by ; << {res_concat} >> Extract only the final equation and return the most common equation as it was provided originally. If there is no common equation return the most likely one. Do not solve the equation, just return the equation."
    self_consistency_prompt_concat = ";".join(self_consistency_prompt)
    messages = [
        ("system", "You are a helpful assistant that extracts the final equation from multiple answers and returns the most common one."),
        ("user", self_consistency_prompt_concat)]
    prompt = ChatPromptTemplate.from_messages(messages)
    model = ChatGroq(model_name=model_name)
    chain = prompt | model
    return chain.invoke({}).content
    


#%%
def main():
    api_key = get_api_key()
    #model_name = 'llama-3.3-70b-versatile'
    model_name = "llama-3.1-8b-instant"
    os.environ["GROQ_API_KEY"] = api_key
    
    user_prompt = "The goal of the Game of 24 is to use the four arithmetic operations (addition, subtraction, multiplication, and division) to combine four numbers and get a result of 24. The numbers are 3, 4, 6, and 8. It is mandatory to use all four numbers. Please check the final equation for correctness. Hints: Identify the basic operations, Prioritize multiplication and division, Look for combinations that make numbers divisible by 24, Consider order of operations, Use parentheses strategically, Practice with different number combinations"

    
    try:
        #res = chain_of_thought_prompting(prompt=user_prompt, model_name=model_name)
        res = self_consistency_cot(prompt=user_prompt, model_name=model_name, n=5)
        pprint(res)
    except Exception as e:
        print(f"Fehler bei der Anfrage: {e}")


if __name__ == "__main__":
    main()# %%

# %%
