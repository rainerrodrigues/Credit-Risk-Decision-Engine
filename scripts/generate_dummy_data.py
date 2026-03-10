import pandas as pd
import numpy as np
import os
import shutil

# Ensuring that the data directory exists in the GitHub Action runner
os.makedirs("data", exist_ok=True)
SAMPLE_DATA_PATH = "tests/test_sample.csv"

print("Generating mock data from test sample...")

# Using the golden sample as the "reference" (training) data
shutil.copy(SAMPLE_DATA_PATH, "data/reference.csv")

# Loading the sample, tweak one column to simulate drift, and save as "current" data
current_data = pd.read_csv(SAMPLE_DATA_PATH)

# Let's forcefully shift a numeric column to trigger EvidentlyAI's drift detection
# Checking the columns and pick a numeric one, e.g., 'loan_amnt' or 'annual_inc'
if 'annual_inc' in current_data.columns:
    current_data['annual_inc'] = current_data['annual_inc'] * 2 

current_data.to_csv("data/current.csv", index=False)

print("✅ Mock data generation complete.")
