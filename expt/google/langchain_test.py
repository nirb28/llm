import sys
# Chat completions API
from langchain_openai import ChatOpenAI
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
MODEL_ID = "meta/llama3-405b-instruct-maas"  # @param {type:"string"} ["meta/llama3-405b-instruct-maas"]

functions = [
    {
        "name": "getWeather",
        "description": "Retrieve real-time weather information/data about a particular location/place",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "the exact location whose real-time weather is to be determined",
                },

            },
            "required": ["location"]
        },
    }
]

llm = ChatOpenAI(
    base_url=f"https://{MODEL_LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{MODEL_LOCATION}/endpoints/openapi/chat/completions?",
    api_key=credentials.token,
    model=MODEL_ID,
    functions=functions
)



template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

llm_chain = prompt | llm

question1 = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

question2 = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]

ai_msg = llm_chain.invoke(question1)
print(ai_msg)