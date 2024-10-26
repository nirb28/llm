import sys, os
# Chat completions API
from langchain_openai import ChatOpenAI
from google.auth import default, transport
from langchain import PromptTemplate
# Build
from langchain_openai import AzureChatOpenAI
from vertexai.preview import rag
from dotenv import load_dotenv

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

load_dotenv()
AZURE_OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT=os.getenv("AZURE_OPENAI_ENDPOINT")

llm = AzureChatOpenAI (
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    functions=functions,
    model="gpt-35-turbo"
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

question3 = "How is the weather in plainsboro?"

ai_msg = llm_chain.invoke(question1)
print(ai_msg)

