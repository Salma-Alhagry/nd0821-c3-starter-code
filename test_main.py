import sys
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}
    
def test_post_0():
    request_data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    
    response = client.post("/model", json = request_data)
    assert response.status_code == 200
    assert response.json() == "<=50K"
    
    
def test_post_1():
    request_data = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 14084,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States",
    }
    
    response = client.post("/model", json = request_data)
    assert response.status_code == 200
    assert response.json() == ">50K"