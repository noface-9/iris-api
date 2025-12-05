# import your FastAPI app
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import app
from fastapi.testclient import TestClient
import pandas as pd
import os
import sys


client = TestClient(app)


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Welcome" in resp.json().get("message", "")


def test_predict_single():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    # model output may vary — assert keys exist and prob in [0,1]
    assert "prediction" in body
    assert "probability" in body
    assert 0.0 <= float(body["probability"]) <= 1.0


def test_batch_prediction_csv(tmp_path):
    # create a small CSV file similar to iris features
    df = pd.DataFrame(
        {
            "sepal_length": [5.1, 6.0],
            "sepal_width": [3.5, 2.9],
            "petal_length": [1.4, 4.5],
            "petal_width": [0.2, 1.5],
        }
    )
    df.columns = df.columns.str.replace("_", "-")
    csv_content = df.to_csv(index=False)
    files = {"file": ("test.csv", csv_content, "text/csv")}
    # send raw bytes as your endpoint expects file: bytes = File(...)
    resp = client.post("/files/", files=files)
    assert resp.status_code == 200
    # response is JSON from DataFrame — ensure keys exist
    data = resp.json()
    # Expect list of rows —
    # or depending on framework conversion
    # your result shape may differ
    first_key = next(iter(data.keys()))
    assert len(data[first_key]) == 2
