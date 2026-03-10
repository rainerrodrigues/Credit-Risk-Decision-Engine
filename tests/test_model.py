import pytest
import joblib
import pandas as pd
import os

MODEL_PATH = "Models/optimized_credit_risk_model.joblib"
SAMPLE_DATA_PATH = "tests/test_sample.csv"

@pytest.fixture
def model():
    assert os.path.exists(MODEL_PATH), "Model path not found!"
    return joblib.load(MODEL_PATH)

def test_model_prediction_output(model):
    # Creating dummy data with the exact columns your model expects
    assert os.path.exists(SAMPLE_DATA_PATH), "Test sample data not found!"
    sample_data = pd.read_csv(SAMPLE_DATA_PATH)
    #dummy_data = pd.DataFrame([{
     #   "age": 35, 
      #  "income": 65000, 
       # "loan_amount": 10000, 
       # "credit_history_length": 5
    # }])
    single_row = sample_data.iloc[[0]]
    
    predictions = model.predict(single_row)
    
    # Checking that a prediction was made
    assert len(predictions) == 1
    # Checking that the prediction is binary (0 = approve, 1 = reject)
    assert predictions[0] in [0, 1] 

def test_model_handles_missing_values(model):
    # Some models fail gracefully, others crash. Test your engine's logic here.
    pass
