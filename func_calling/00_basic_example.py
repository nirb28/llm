import requests
import json,os
from dotenv import load_dotenv

load_dotenv()
# Define the OpenAI endpoint and API key
api_url = "https://instance-ds-openai.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2023-03-15-preview"
#api_url = "https://api.groq.com/openai/v1/chat/completions"
api_key = os.getenv("AZURE_OPENAI_API_KEY")
#api_key = os.getenv("GROQ_API_KEY")
#model = "llama-3.1-70b-versatile"
model = "gpt-35-turbo"

# Define the function for getting current weather
def get_current_weather(city):
    return f"The current weather in {city} is sunny."

# Define the tool
tools = [{
    'type': 'function',
    'function': {
        'name': 'get_current_weather',
        'description': 'This tool simulates checking the weather by randomly selecting from three possible outcomes: sunny, cold, or rainy.',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {
                    'type': 'string',
                    'description': 'The name of the city',
                },
            },
            'required': ['city'],
        },
    },
}]

# Define the request payload
payload = {
    "model": model, #mistral-nemo:12b-instruct-2407-q8_0
    "messages": [
        {"role": "system", "content": "You are a smart AI assistant. You are a master at understanding what a customer wants and utilize available tools only if you have to."},
        {"role": "user", "content": "What is the weather in Woodbury MN?"}
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
        if call['type'] == 'function' and call['function']['name'] == 'get_current_weather':
            city = json.loads(call['function']['arguments'])['city']
            weather = get_current_weather(city)
            print(f"Weather in {city}: {weather}")
else:
    print(f"Error: {response.status_code}, {response.text}")
