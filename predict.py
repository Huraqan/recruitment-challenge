# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split#, GridSearchCV
# from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score
# import sklearn
import joblib
import pandas as pd

def predict(inputs):
    inputs = dict(inputs)
    
    for k in inputs.keys():
        inputs[k] = [inputs[k]]
    data = pd.DataFrame(inputs)
    
    # Load and unpack the model
    artifacts = joblib.load("artifacts_rf_better.joblib")
    
    num_features = artifacts["features"]["num_features"]
    cat_features = artifacts["features"]["cat_features"]
    enc = artifacts["enc"]
    model = artifacts["model"]
    
    data = data[cat_features + num_features]
    
    # Apply imputer, scaler and encoder on data
    data_cat = enc.transform(data[cat_features])
    
    # Combine the numerical and one-hot encoded categorical columns
    data = pd.concat(
        [
            pd.DataFrame(data_cat, columns=enc.get_feature_names_out()),
            data[num_features].reset_index(drop=True),
        ],
        axis=1,
    )
    
    prediction = {"pred": model.predict(data)[0]}
    return prediction