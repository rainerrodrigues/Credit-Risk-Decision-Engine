import os
import shutil

# Ensuring the data directory exists
os.makedirs("data", exist_ok=True)
SAMPLE_DATA_PATH = "tests/test_sample.csv"

print("Generating clean mock data from test sample...")

# Copying the golden sample to both reference and current data
# This ensures they are identical, resulting in 0 data drift.
shutil.copy(SAMPLE_DATA_PATH, "data/reference.csv")
shutil.copy(SAMPLE_DATA_PATH, "data/current.csv")

print("✅ Clean mock data generation complete. No drift should be detected.")
