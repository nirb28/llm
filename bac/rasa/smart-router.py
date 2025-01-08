import re
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load regex rules from config file
with open('config.json') as config_file:
    config = json.load(config_file)
    rules = config['rules']

@app.route('/')
def home():
    return "Welcome to the Home Page!"

@app.route('/route_message', methods=['POST'])
def route_message():
    data = request.get_json()
    message = data.get('message', '')
    lob = data.get('lob', None)

    for rule in rules:
        if re.search(rule['pattern'], message):
            return rule['destination']
    
    return lob if lob is not None else "GENERAL"

@app.route('/about')
def about():
    return "This is the About Page."

if __name__ == '__main__':
    app.run(debug=True)