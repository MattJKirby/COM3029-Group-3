import os
from src.logger import Logger
from src.model import Model
from flask import Flask, make_response, request
app = Flask(__name__)

logger = Logger()
model = Model(os.getenv('MODEL_PATH'))

@app.route("/", methods = ['GET'])
def home():
    return "To predict, provide an input! \n For example --> http://localhost:8080/web-service/predict?input=help"

@app.route("/status", methods = ['GET'])
def status():
    response = make_response('OK', 200)
    user_agent = request.headers.get('User-Agent')
    logger.log(f"Request: GET /status - User-Agent: {user_agent} - Response: {response._status_code}")
    return response

@app.route("/predict", methods = ['GET'])
def predict():
    user_agent = request.headers.get('User-Agent')
    logger.log(f"Request: GET /predict - User-Agent: {user_agent}")
    input = request.args.get('input')

    if(input and len(input) > 0):
        prediction = model.predict(input)
        logger.log(f"Predicting emotion for input: {input} - Prediction: {prediction}")
        return make_response(prediction, 200)
    else:
        return make_response("Please provide an input! \n For example --> http://localhost:8080/api/web-service/predict?input=help", 200)
