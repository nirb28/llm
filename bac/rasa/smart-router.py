import re
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Home Page!"

@app.route('/route_message', methods=['POST'])
def route_message():
    data = request.get_json()
    message = data.get('message', '')
    lob = data.get('lob', None)

    if re.search(r'\bMALTS\b', message, re.IGNORECASE):
        return "MALTS"
    elif re.search(r'\bMSaaS\b', message, re.IGNORECASE):
        return "MSaaS"
    elif re.search(r'\bDQ4D\b', message, re.IGNORECASE):
        return "DQ4D"
    else:
        return lob if lob is not None else "GENERAL"

@app.route('/about')
def about():
    return "This is the About Page."

if __name__ == '__main__':
    app.run(debug=True)