"""App.py"""
import json
import os
import sys
import uuid
from flask import Flask, jsonify, request
from flask_bcrypt import Bcrypt
from flask_cors import CORS
import query_response

DOMAINS_ALLOWED = "*"  # You can restrict only for your sites here
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": DOMAINS_ALLOWED}})
# CORS(app)
# Get the environmental variables
app.config["DEBUG"] = os.environ.get("FLASK_DEBUG")
bcrypt = Bcrypt(app)


@app.route('/', methods=['GET'])
def home():
    """
    Returns a message with the Python version.

    Returns:
        str: A message with the Python version.
    """
    message = "Python version: " + sys.version
    return message


@app.route('/getQueryResponse', methods=['POST'])
def query_response():
    """
    Endpoint to get a response from an avatar service.

    Request Body:
        JSON: Contains the message for the avatar service.

    Returns:
        str: The response from the avatar service.
    """
    try:
        data = request.json
        input_text = data['query']
        session_id = data.get()('session_id', uuid.uuid4())
        if input_text != '':
            response = query_response.process_query(input_text)
            response_data = {"message": response, "session_id": session_id}
            return jsonify({'response': response_data, "success": True}), 200
        else:
            return jsonify({'response': 'Query Cannot be blank', "success": False}), 401
    except Exception as err:
        return jsonify({'response': err, "success": False}), 500


if __name__ == '__main__':
    app.run()
