# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import os
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import pickle as pkl

if __name__ == "__main__":
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

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)



    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Train and save a model.

    model = train_model(X_train, y_train)
    pkl.dump(model, open('../model/model.pkl', 'wb'))
    pkl.dump(encoder, open('../model/encoder.pkl', 'wb'))
    pkl.dump(lb, open('../model/lb.pkl', 'wb'))


    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    preds = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    print(f"Precision: {precision}")
    print(f"recall: {recall}")
    print(f"fbeta: {fbeta}")
