from src.logger import Logger
from flask import Flask, make_response, request
app = Flask(__name__)

logger = Logger()

@app.route("/")
def home():
    return "Hello, Flask!"

@app.route("/status")
def status():
    response = make_response('OK', 200)
    user_agent = request.headers.get('User-Agent')
    logger.log(f"Request: GET /status - User-Agent: {user_agent} - Response: 200")
    return response
