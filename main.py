from fastapi import FastAPI, File
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from io import StringIO

# import requests

app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


class IrisSpecies(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
async def root():
    return {"message": "Welcome to the Iris Species Prediction API!"}


@app.post("/predict")
async def predict_species(iris: IrisSpecies):
    data = iris.model_dump()

    # Create DataFrame with column names that match training
    df = pd.DataFrame([data])
    df = df.rename(
        columns={
            "sepal_length": "sepal-length",
            "sepal_width": "sepal-width",
            "petal_length": "petal-length",
            "petal_width": "petal-width",
        }
    )
    prediction = model.predict(df)
    probability = model.predict_proba(df)

    return {"prediction": prediction[0], "probability": float(np.max(probability))}


# @app.post("/predict")
# async def predict_species(iris: IrisSpecies):
#     data = iris.dict()
#     data_in = [[data['sepal_length'],
#     data['sepal_width'],
#     data['petal_length'],
#     data['petal_width']]]
#     print(data_in)
#     endpoint = "http://localhost:1234/invocations"
#     inference_request = {
#         "inputs": data_in
#     }
#     print(inference_request)
#     response = requests.post(endpoint, json=inference_request)
#     print(response)
#     return {
#         "predicted_species": response.text,}


@app.post("/files/")
async def batch_prediction(file: bytes = File(...)):
    s = str(file, "utf-8")
    data = StringIO(s)
    df = pd.read_csv(data)
    df = df.rename(
        columns={
            "sepal_length": "sepal-length",
            "sepal_width": "sepal-width",
            "petal_length": "petal-length",
            "petal_width": "petal-width",
        }
    )
    # # STANDARDIZE column names
    # df.columns = df.columns.str.strip().str.lower().str.replace("-", "_")

    # Ensure correct order
    expected = ["sepal-length", "sepal-width", "petal-length", "petal-width"]
    df = df[expected]

    prediction = model.predict(df)
    probability = model.predict_proba(df)

    result = pd.DataFrame(
        {
            "Pred_Class": prediction,
            "Prob_being_Iris-setosa": probability[:, 0],
            "Prob_being_Iris-versicolor": probability[:, 1],
            "Prob_being_Iris-virginica": probability[:, 2],
        }
    )

    return result


# @app.post("/files/")
# async def batch_prediction(file: bytes = File(...)):
#     s = str(file, 'utf-8')
#     data = StringIO(s)
#     df = pd.read_csv(data)
#     lst = df.values.tolist()
#     endpoint = "http://localhost:1234/invocations"
#     inference_request = {
#         "inputs": lst
#     }
#     response = requests.post(endpoint, json=inference_request)
#     print(response.text)

#     return response.text
