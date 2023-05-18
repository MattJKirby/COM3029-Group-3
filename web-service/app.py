from flask import Flask, make_response
app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Flask!"

@app.route("/status")
def status():
    response = make_response('OK', 200)
    return response
