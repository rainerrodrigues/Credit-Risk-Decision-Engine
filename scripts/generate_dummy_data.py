import pandas as pd
import numpy as np
import os

# Ensuring that the data directory exists in the GitHub Action runner
os.makedirs("data", exist_ok=True)

# Generating Reference Data (Simulating our original training data)
reference_data = pd.DataFrame({
    "age": np.random.randint(20, 60, 100),
    "income": np.random.randint(30000, 120000, 100),
    "loan_amount": np.random.randint(5000, 50000, 100),
    "credit_history_length": np.random.randint(1, 20, 100)
})
reference_data.to_csv("data/reference.csv", index=False)
print("Created data/reference.csv")

# Generating Current Data (Simulating new data from our FastAPI server)
# We intentionally shift the 'income' distribution higher to trigger data drift
current_data = pd.DataFrame({
    "age": np.random.randint(20, 60, 100),
    "income": np.random.randint(80000, 200000, 100), 
    "loan_amount": np.random.randint(5000, 50000, 100),
    "credit_history_length": np.random.randint(1, 20, 100)
})
current_data.to_csv("data/current.csv", index=False)
print("Created data/current.csv")

print("✅ Mock data generation complete.")
