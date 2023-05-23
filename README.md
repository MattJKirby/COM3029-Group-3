# COM3029-Group-3

## How To Run The Docker Compose Application

- Clone the repo
- Open the repo in terminal
- Run ```docker compose up --build``` terminal command within the project root
- Navigate to ```http://localhost:8080/api/web-service/``` within the web browser


## Build Script Setup
Running ```build-script.py``` requires the installation of certain libraries. Please run the following ```pip``` command:

```pip install tensorflow nltk language_tool_python transformers datasets```

The resulting model will be output at ```models/model``` and takes the form of a folder called ```model```