# Iris Classification API
A simple machine learning classification project using the classic Iris dataset, built with Random Forest, deployed as a FastAPI application, and containerized using Docker.
This project demonstrates a full end-to-end workflow:
 * Building a model
 * Saving it with joblib
 * Creating a prediction API (FastAPI) 
 * Running it locally
 * Containerizing with Docker

## Iris Dataset Reference
The Iris dataset is a classic dataset in machine learning introduced by Ronald Fisher (1936) and widely used for classification tasks.

## Purpose of This Project
This project is designed for:
 * Understanding model → API → Docker deployment flow
 * Practicing MLOps basics
 * Building a portfolio project
 * Learning how to containerize ML models

## Project Overview
This project uses the Iris dataset (3 classes: Setosa, Versicolor, Virginica) to train a RandomForestClassifier.
The trained model is served through a FastAPI endpoint where users can send flower measurements and receive predicted species.

## Model Used
Algorithm: Random Forest Classifier
Library: scikit-learn
Features:
 * sepal_length
 * sepal_width
 * petal_length
 * petal_width

## Model is saved as:
```bash
model.pkl
```

## Tech Stack
|Component|Tool|
|:---:|:---:|
|Model|Random Forest Classifier|
|API Framework|FastAPI|
|Server|Uvicorn|
|Packaging|Docker|
|Language|Python 3.x|

## Project Structure
```bash
--- .dockerignore         # Files to ignore in Docker
--- gitignore            # Files to ignore in git
--- main.py               # FastAPI application
--- model.pkl             # Not saved in Git - has pkl model
--- requirements.txt      # Dependencies
--- Dockerfile            # Docker configuration
--- README.md             # Documentation
--- iris.csv              # Input used for training
--- iris_predictor.csv    # Input used for testing
```


Running the API Locally
1. Create & activate your environment (optional)
```bash
conda create -n iris_api python=3.10
conda activate iris_api
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Start the FastAPI server
```bash
uvicorn main:app --reload
```
4. Open the API docs
FastAPI provides automatic documentation:
```bash
http://127.0.0.1:8000/docs
```
```bash
http://127.0.0.1:8000/redoc
```

## Example Prediction Request (JSON)
```bash
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

## Running with Docker
1. Build the Docker image
```bash
docker build -t iris-api
```
2. Run the container
```bash
docker run -p 8000:8000 iris-api
```
3. Access the API
```bash
http://localhost:8000/docs
```

## Contributions
Feel free to fork this project, raise issues, or submit pull requests.

## If you find this project helpful
Please give the repo a ⭐ on GitHub — it motivates me to build more!
