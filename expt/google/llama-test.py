import sys
# Chat completions API
import openai
from google.auth import default, transport
from langchain import PromptTemplate
# Build
from langchain_openai import ChatOpenAI
from vertexai.preview import rag

# gcloud auth print-access-token

PROJECT_ID = "llm-playarea"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

BUCKET_NAME = "bucket-ds-llm"  # @param {type:"string"}
BUCKET_URI = f"gs://{BUCKET_NAME}"

# before running this run: gcloud auth application-default login
# This writes the authentication info to C:\Users\dalje\AppData\Roaming\gcloud\application_default_credentials.json
credentials, _ = default()
auth_request = transport.requests.Request()
credentials.refresh(auth_request)

MODEL_LOCATION = "us-central1"

client = openai.OpenAI(
    base_url=f"https://{MODEL_LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{MODEL_LOCATION}/endpoints/openapi/chat/completions?",
    api_key=credentials.token,
)

MODEL_ID = "meta/llama3-405b-instruct-maas"  # @param {type:"string"} ["meta/llama3-405b-instruct-maas"]
response = client.chat.completions.create(
    model=MODEL_ID, messages=[{"role": "user", "content": "Hello, Llama 3.1!"}]
)
print(response.choices[0].message.content)

temperature = 1.0  # @param {type:"number"}
max_tokens = 50  # @param {type:"integer"}
top_p = 1.0  # @param {type:"number"}
stream = True  # @param {type:"boolean"}

response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[
        {"role": "user", "content": "What is Vertex AI?"},
        {"role": "assistant", "content": "Sure, Vertex AI is:"},
    ],
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=top_p,
    stream=stream,
)
