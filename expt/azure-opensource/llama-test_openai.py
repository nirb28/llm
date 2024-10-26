import sys
# Chat completions API
import openai
from google.auth import default, transport

url = 'https://Meta-Llama-3-1-70B-Instruct-gvmr.eastus2.models.ai.azure.com/v1/chat/completions'
# Replace this with the primary/secondary key, AMLToken, or Microsoft Entra ID token for the endpoint
api_key = 'hsGeJQRJPy1NYqL9bwCDRBPs4WBhzZo9'

client = openai.OpenAI(
    base_url=url,
    api_key=api_key,
)
response = client.chat.completions.create(
    model="Meta-Llama-3.1-70B-Instruct",
    messages=[{"role": "user", "content": "Hello, Llama 3.1!"}]
)
print(response.choices[0].message.content)

temperature = 1.0  # @param {type:"number"}
max_tokens = 50  # @param {type:"integer"}
top_p = 1.0  # @param {type:"number"}
stream = True  # @param {type:"boolean"}

response = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "What is Vertex AI?"},
        {"role": "assistant", "content": "Sure, Vertex AI is:"},
    ],
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=top_p,
    stream=stream,
)
