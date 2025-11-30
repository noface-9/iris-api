Iris Classification API
A simple machine learning classification project using the classic Iris dataset, built with Random Forest, deployed as a FastAPI application, and containerized using Docker.
This project demonstrates a full end-to-end workflow:
  Building a model
  Saving it with joblib
  Creating a prediction API (FastAPI) 
  Running it locally
  Containerizing with Docker

Project Overview
This project uses the Iris dataset (3 classes: Setosa, Versicolor, Virginica) to train a RandomForestClassifier.
The trained model is served through a FastAPI endpoint where users can send flower measurements and receive predicted species.

Model Used
Algorithm: Random Forest Classifier
Library: scikit-learn
Features:
sepal_length
sepal_width
petal_length
petal_width
