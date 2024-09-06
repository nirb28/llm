import sys, os
# Chat completions API
from langchain_openai import ChatOpenAI
from google.auth import default, transport
from langchain import PromptTemplate
# Build
from langchain_openai import ChatOpenAI
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
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
llm = ChatOpenAI(
    base_url=f"https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
    functions=functions,
    model="llama3-8b-8192"
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