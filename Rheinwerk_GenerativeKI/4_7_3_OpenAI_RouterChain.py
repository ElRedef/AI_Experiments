#%% packages
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.utils.math import cosine_similarity
from dotenv import load_dotenv


load_dotenv('4_2_2_groq.env')
print("OPENAI Router Chain")
print(os.getenv('OPENAI_API_KEY'))

# Initialize OpenAI
embeddings = OpenAIEmbeddings()

MODEL_NAME = "gpt-4o-mini"
model = ChatOpenAI(model_name=MODEL_NAME, temperature=0, openai_api_key=os.getenv('OPENAI_API_KEY'))

#%% Prompts definieren
# Templates
template_math = "Solve the following math problem: {user_input}, state that you are a math agent"
template_music = "Suggest a song for the user: {user_input}, state that you are a music agent"
template_history = "Provide a history lesson for the user: {user_input}, state that you are a history agent" 

prompt_math = ChatPromptTemplate.from_messages([("system", template_math), ("human", "{user_input}")])
chain_math = prompt_math | model | StrOutputParser()
 
prompt_music = ChatPromptTemplate.from_messages([("system", template_music),("human", "{user_input}")])
chain_music = prompt_music | model | StrOutputParser()
 
prompt_history = ChatPromptTemplate.from_messages([("system", template_history),("human", "{user_input}")])
chain_history = prompt_history | model | StrOutputParser()


# %% Chains & Embeddings
chains = [chain_math, chain_music, chain_history] 
chain_embeddings = embeddings.embed_documents(["math", "music", "history"])


# %% Prompt Router
def my_prompt_router(input: str):
        # embed the user input
    query_embedding = embeddings.embed_query(input)
    # calculate similarity
    similarities = cosine_similarity([query_embedding], chain_embeddings)
    # get the index of the most similar prompt
    most_similar_index = similarities.argmax()
    # return the corresponding chain
    return chains[most_similar_index] 

def answer_query(input: str):
    chain = my_prompt_router(input)
    print(chain.invoke({"user_input": input}))

#%% Test1
query = "Von wem ist das Lied MFG?"
answer_query(query)


# %% Test2
query = "Was sind die Hauptthemen in Beethovens 9. Sinfonie?"
answer_query(query)

# %% # Test3
query = "Was ist die Ableitung von x^2 + 3x + 2?"
answer_query(query)

# %% Test4
query = "Was ist die Hauptstadt von Frankreich?"
answer_query(query)

# %% Test5
query = "Was war 333 vor Christus"
answer_query(query)

# %%
query = "Was war 333 nach Christus"
answer_query(query)

# %%
