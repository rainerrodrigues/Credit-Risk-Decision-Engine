import pytest
import joblib
import pandas as pd
import os

MODEL_PATH = "Models/optimized_credit_risk_model.joblib"

@pytest.fixture
dev model():
    assert os.path.exists(MODEL_PATH), "Model path not found!"
    return joblib.load(MODEL_PATH)

def test_model_prediction_output(model):
    # Creating dummy data with the exact columns your model expects
    dummy_data = pd.DataFrame([{
        "age": 35, 
        "income": 65000, 
        "loan_amount": 10000, 
        "credit_history_length": 5
    }])
    
    predictions = model.predict(dummy_data)
    
    # Checking that a prediction was made
    assert len(predictions) == 1
    # Checking that the prediction is binary (0 = approve, 1 = reject)
    assert predictions[0] in [0, 1] 

def test_model_handles_missing_values(model):
    # Some models fail gracefully, others crash. Test your engine's logic here.
    pass
