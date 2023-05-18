from src.logger import Logger
from flask import Flask, make_response
app = Flask(__name__)

logger = Logger()

@app.route("/")
def home():
    return "Hello, Flask!"

@app.route("/status")
def status():
    response = make_response('OK', 200)
    logger.log("/status endpoint called")
    return response
