FROM python:3.10


WORKDIR /app
COPY ./requirements.txt /app


ENV PYTHONPATH=${PYTHONPATH}:${PWD} 

RUN pip install -r requirements.txt
RUN pip install --no-cache-dir tensorflow==2.12.0

COPY . .

RUN unzip models/model_best/model-placehoder.zip

EXPOSE 5000

ENV LISTEN_PORT=5000

ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000
ENV FLASK_APP="app.py"
ENV MODEL_BEST_PATH="models/model_best"
ENV MODEL_PATH="models/model"

CMD [ "flask", "run" ]
