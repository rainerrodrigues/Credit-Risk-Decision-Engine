import pandas as pd
import joblib
import shap
import numpy as np
import os
from google import genai 
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
MODEL_PATH = 'Models/optimized_credit_risk_model.joblib'
LOWER_THRESHOLD = 0.20  
UPPER_THRESHOLD = 0.60  
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"      

def get_column_names(preprocessor):
    """Extracts feature names from the pipeline's ColumnTransformer."""
    num_features = preprocessor.transformers_[0][2].tolist()
    cat_transformer = preprocessor.named_transformers_['cat']
    cat_features_orig = preprocessor.transformers_[1][2]
    cat_features_transformed = cat_transformer.get_feature_names_out(cat_features_orig).tolist()
    return num_features + cat_features_transformed

def make_decision(probability):
    if probability < LOWER_THRESHOLD:
        return "AUTOMATIC ACCEPT", "Applicant presents low default risk."
    elif probability > UPPER_THRESHOLD:
        return "AUTOMATIC REJECT", "Applicant exceeds risk tolerance thresholds."
    else:
        return "MANUAL REVIEW", "Moderate risk segment; assessment by Credit Officer required."

def generate_human_explanation(model_results, client):
    prompt = f"""
    Acting as a Senior Credit Risk Officer, provide a concise 2-3 sentence internal summary for this application:
    - Decision: {model_results['decision']}
    - Probability of Default: {model_results['probability_of_default'] * 100:.2f}%
    - Key Drivers: {', '.join(model_results['top_drivers'])}
    
    The summary should be professional, data-driven, and explain why the applicant was placed in this specific decision segment.
    """
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Narrative generation unavailable: {str(e)}"

def predict_with_explanation(input_data):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    pipeline = joblib.load(MODEL_PATH)
    classifier = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    df = pd.DataFrame([input_data])
    df = df[pipeline.feature_names_in_]

    prob_default = pipeline.predict_proba(df)[0][1]
    decision, segment_logic = make_decision(prob_default)

    X_transformed = preprocessor.transform(df)
    feature_names = get_column_names(preprocessor)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)

    instance_shap = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

    contributions = sorted(zip(feature_names, instance_shap), key=lambda x: abs(x[1]), reverse=True)
    detailed_reasons = []
    
    for feat_name, shap_val in contributions[:3]:
        impact = "Decreased Risk" if shap_val < 0 else "Increased Risk"
        
        if feat_name in input_data:
            val = input_data[feat_name]
            detailed_reasons.append(f"{feat_name}: {val} ({impact})")
        
        else:
            found = False
            for original_col in input_data.keys():
                if feat_name.startswith(original_col + "_"):
                    val = input_data[original_col]
                    detailed_reasons.append(f"{original_col}: {val} ({impact})")
                    found = True
                    break
            if not found:
                detailed_reasons.append(f"{feat_name} ({impact})")

    results = {
        "probability_of_default": round(float(prob_default), 4),
        "decision": decision,
        "top_drivers": detailed_reasons
    }

    client = genai.Client(api_key=GEMINI_API_KEY)
    results["narrative"] = generate_human_explanation(results, client)

    return results

if __name__ == "__main__":
    test_df_path = 'data/data_imputation_test.csv'
    if os.path.exists(test_df_path):
        df = pd.read_csv(test_df_path)
        sample_input = df.iloc[10].to_dict()
        
        res = predict_with_explanation(sample_input)
        
        print("\n" + "="*60)
        print("                CREDIT RISK ASSESSMENT REPORT")
        print("="*60)
        print(f"DECISION:        {res['decision']}")
        print(f"PD SCORE:        {res['probability_of_default'] * 100:.2f}%")
        print("-" * 60)
        print("CRITICAL RISK DRIVERS:")
        for reason in res['top_drivers']:
            print(f" • {reason}")
        print("-" * 60)
        print(f"EXECUTIVE SUMMARY:\n{res['narrative']}")
        print("="*60)
    else:
        print("Data file not found. Please check paths.")