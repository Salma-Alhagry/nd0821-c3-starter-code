import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

import pickle as pkl

import pytest
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference


@pytest.fixture()
def input_data():
    file_dir = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(file_dir, "data/census_cleaned.csv"))
    
    train, test = train_test_split(data, test_size=0.20)
    
    return train, test

def test_process_data(input_data):
    
    train, test = input_data
    
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    
def test_inference(input_data):
    
    train, test = input_data
    
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    
    model = train_model(X_train, y_train)

    preds = inference(model, X_train)
    
    assert len(preds) == len(y_train)
    
def test_compute_model_metrics(input_data):
    
    train, test = input_data
    
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]
    
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    precision, recall, fbeta = compute_model_metrics(y_train, preds)
    
    assert precision <= 1.0
    assert recall <= 1.0
    assert fbeta <= 1.0
   
    