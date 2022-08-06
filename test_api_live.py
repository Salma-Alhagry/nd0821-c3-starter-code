import requests

url = "https://salma-udacity-mlops-p3.herokuapp.com/"

response = requests.get(url)
print(f"Status code: {response.status_code}")

response = requests.post(url, json={
    "summary": "Person",
    "value":{
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
})
print(f"Status code: {response.status_code}")