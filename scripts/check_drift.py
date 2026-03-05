import pandas as pd
import sys
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset

def run_drift_checks():
    print("Loading data...")
    #I will adjust paths as needed
    reference_data = pd.read_csv("data/reference.csv")
    current_data = pd.read_csv("data/current.csv")

    print("Running EvidentlyAI Data Drift Test Suite...")
    data_drift_suite = TestSuite(tests=[
        DataDriftTestPreset(),
    ])
    
    data_drift_suite.run(reference_data=reference_data, current_data=current_data)
    
    # Saving the report as HTML so you can download it from GitHub later
    data_drift_suite.save_html("drift_report.html")
    
    # Checking if any tests failed
    summary = data_drift_suite.as_dict()
    if not summary["summary"]["all_passed"]:
        print("❌ DATA DRIFT DETECTED! Pipeline will fail.")
        sys.exit(1) # Exits with error code 1, which fails the GitHub Action
    else:
        print("✅ No data drift detected. Tests passed.")
        sys.exit(0)

if __name__ == "__main__":
    run_drift_checks()
