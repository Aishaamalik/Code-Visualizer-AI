import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",   
    temperature=0.7,
    max_tokens=200
)

# Ask something
response = llm.invoke("Write a short poem about coding in Python.")
print(response.content)