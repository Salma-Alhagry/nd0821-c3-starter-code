# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import os
import pandas as pd
from ml.data import process_data
from ml.model import compute_model_metrics, inference
import pickle as pkl

# Add code to load in the data.
file_dir = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(file_dir, "data/census_cleaned.csv"))

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
def slice_perform(cat_featue):
    model = pkl.load(open('model/model.pkl', 'rb'))
    encoder = pkl.load(open('model/encoder.pkl', 'rb'))
    lb = pkl.load(open('model/lb.pkl', 'rb'))

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

    with open("slice_output.txt", "w") as file:
        #for feature in cat_features:
        for slice_value in data[cat_featue].unique():
            slice_df = test[test[cat_featue] == slice_value]

            X_test, y_test, _, _ = process_data(
                slice_df, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
            )

            preds = inference(model, X_test)

            precision, recall, fbeta = compute_model_metrics(y_test, preds)

            file.write(f"Precision: {precision}\n")
            file.write(f"recall: {recall}\n")
            file.write(f"fbeta: {fbeta}\n")
            file.write("\n")
            
if __name__ == "__main__":
    slice_perform(cat_features[0]) 