import sys
# Chat completions API
import openai, os
from dotenv import load_dotenv
from google.auth import default, transport
from langchain import PromptTemplate
# Build
from langchain_openai import ChatOpenAI
from vertexai.preview import rag

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

client = openai.OpenAI(
    base_url=f"https://api.groq.com/openai/v1/chat/completions?",
    api_key=GROQ_API_KEY,
)

MODEL_ID = "llama3-8b-8192"
response = client.chat.completions.create(
    model=MODEL_ID, messages=[{"role": "user", "content": "Hello, Llama 3.1!"}]
)
print(response.choices[0].message.content)

