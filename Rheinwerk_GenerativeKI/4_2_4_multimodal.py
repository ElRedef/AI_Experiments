
#%% packages
print("Hello Groq Multimodal\n")
from groq import Groq
from dotenv import load_dotenv
import os
import base64
import sys

#%% Variablen

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
IMAGE_PATH = "D:\\Github\\AI_Experiments2\\Rheinwerk_GenerativeKI\\bild.JPG"
USER_PROMPT = "What is shown in this image? Answer in one sentence. Please answer in German."

#%% Helfer Funktionen definieren 

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Fehler: Bilddatei '{image_path}' nicht gefunden.")
        sys.exit(1)


def get_api_key():
    # Lade explizit die Datei 4_2_2_groq.env
    load_dotenv('4_2_2_groq.env')
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("Fehler: GROQ_API_KEY nicht in 4_2_2_groq.env gefunden.")
        sys.exit(1)
    return api_key


# %%

def main():
    
    api_key = get_api_key()
    base64_image = encode_image(IMAGE_PATH)
    client = Groq(api_key=api_key)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model=MODEL,
        )
        print("Antwort:", chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"Fehler bei der Anfrage: {e}")

if __name__ == "__main__":
    main()

# %%
