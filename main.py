# Put the code for your API here.

import sys, os
import pandas as pd
import pickle as pkl
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import process_data
from ml.model import inference


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

file_dir = os.path.dirname(__file__)
#data = pd.read_csv(os.path.join(file_dir, "../data/census_cleaned.csv"))
print("change")
model = pkl.load(open(os.path.join(file_dir,'model/model.pkl'), 'rb'))
encoder = pkl.load(open(os.path.join(file_dir,'model/encoder.pkl'), 'rb'))
lb = pkl.load(open(os.path.join(file_dir,'model/lb.pkl'), 'rb'))

class InputData(BaseModel):
    age: int = Field(..., example=35)
    workclass: str = Field(..., example="Never-worked")
    fnlgt: int = Field(..., example=83311)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(alias="education-num", example=7)
    marital_status: str = Field(alias="marital-status", example="Never-married")
    occupation: str = Field(..., example="Craft-repair")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="Black")
    sex: str = Field(..., example="Female")
    capital_gain: int  = Field(alias="capital-gain", example=2961)
    capital_loss: int  = Field(alias="capital-loss", example=2002)
    hours_per_week: int = Field(alias="hours-per-week", example=77)
    native_country: str = Field(alias="native-country", example="Hungary")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/model")
def predict(input_data: InputData):
     
        df_data = pd.DataFrame.from_dict([input_data.dict(by_alias=True)])
        
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
        
        X, y, _, _ = process_data(
            df_data, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb
        )
        
        preds = inference(model, X)
        
        return "<=50K" if preds[0] == 0 else ">50K"
        
