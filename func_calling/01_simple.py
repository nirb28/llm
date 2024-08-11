import requests
import json, os
from weather_tool import get_weather
from dotenv import load_dotenv

load_dotenv()
# Define the OpenAI endpoint and API key
api_url = "https://instance-ds-openai.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2023-03-15-preview"
#api_url = "https://api.groq.com/openai/v1/chat/completions"
api_key = os.getenv("AZURE_OPENAI_API_KEY")
#api_key = os.getenv("GROQ_API_KEY")
#model = "llama-3.1-70b-versatile"
model = "gpt-35-turbo"

# Collect tool definitions from decorated functions
tools = [get_weather.tool_definition]

# Define the request payload
payload = {
    "model": model, #llama3.1:8b-instruct-q8_0
    "messages": [
        {
            "role": "system",
            "content": "You are a smart AI assistant. You are a master at understanding what a customer wants and utilize available tools only if you have to."
         },
        {
            "role": "user", 
            "content": "How hot is it in minneapolis??"
        }
    ],
    "tools": tools
}

# Make the request to the OpenAI API
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
    "api-key": api_key,
}
response = requests.post(api_url, headers=headers, data=json.dumps(payload))

print(response.json())

# Handle the tool response
if response.status_code == 200:
    response_data = response.json()
    tool_calls = response_data.get('choices', [{}])[0].get('message', {}).get('tool_calls', [])
    for call in tool_calls:
        if call['type'] == 'function' and call['function']['name'] == 'get_weather':
            location = json.loads(call['function']['arguments'])['location']
            weather = get_weather(location)
            print(f"Weather in {location}: {weather}")
else:
    print(f"Error: {response.status_code}, {response.text}")
