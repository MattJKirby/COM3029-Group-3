import os
from src.logger import Logger
from src.model import Model
from flask import Flask, make_response, request
app = Flask(__name__)

logger = Logger()
model_best = Model(os.getenv('MODEL_BEST_PATH'))
model = Model(os.getenv('MODEL_PATH'))

@app.route("/", methods = ['GET'])
def home():
    return "To predict, provide an input! \n For example --> http://localhost:8080/api/web-service/predict?input=help"

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
    type = request.args.get('type')

    if(input and len(input) > 0):
        prediction = None
        
        if(type == "new" and os.path.isdir(os.getenv('MODEL_PATH'))):
          prediction = model.predict(input)
          logger.log(f"Predicting emotion for input: {input} - Prediction: {prediction} - Model type: model")
        else:
          prediction = model_best.predict(input)
          logger.log(f"Predicting emotion for input: {input} - Prediction: {prediction} - Model type: best")
          
        return make_response(prediction, 200)
    else:
        return make_response("Please provide an input! \n For example --> http://localhost:8080/api/web-service/predict?input=help", 200)
