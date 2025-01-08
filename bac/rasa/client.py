import requests, os, json

def send_message(message):
    url = 'http://127.0.0.1:5000/route_message'
    payload = {'message': message, 'lob': 'None'}
    headers = {'Content-Type': 'application/json'}

    response = requests.post(url, json=payload, headers=headers)
    print(response.text)

if __name__ == '__main__':
    # Example messages
    #send_message("This is a test message containing MALT.")
    # send_message("This message mentions MSaaS.")
    # send_message("Here we talk about DQ4D.")
    # send_message("No keywords here.")
    print("Current working directory:", os.getcwd())
    # Load regex rules from config file
    with open('config.json') as config_file:
        config = json.load(config_file)
        rules = config['rules']        
        print(rules)