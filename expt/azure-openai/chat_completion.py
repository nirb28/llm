import os
from openai import AzureOpenAI
import json
from dotenv import load_dotenv

load_dotenv()
AZURE_OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT=os.getenv("AZURE_OPENAI_ENDPOINT")

client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY
)

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            #"content": "Who is the CEO of Bank of America?"
            "content": "What is the weather in Plainsboro?",
        }
    ]
)
      
print(completion.to_json())

