import pandas as pd
import joblib
import shap
import numpy as np
import os
from google import genai 
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = 'Models/finalized_credit_model.joblib'
LOWER_THRESHOLD = 0.25  
UPPER_THRESHOLD = 0.60  
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash" 

MOCK_BUREAU_DATA = {
    # Assuming applicant has a clean record unless the credit pull proves otherwise.
    'delinq_2yrs': 0, 
    'pub_rec': 0, 
    'collections_12_mths_ex_med': 0, 
    'acc_now_delinq': 0, 
    'tot_coll_amt': 0, 
    'inq_last_6mths': 0, 

    # 0 would indicate that the applicant just had a bankruptcy or delinquency yesterday. 
    # Hence using 999 to indicate "no history" and avoid skewing the model with extreme values.
    'mths_since_last_delinq': 999.0,
    'mths_since_last_record': 999.0,
    'mths_since_last_major_derog': 999.0,
    'mths_since_recent_bc_dlq': 999.0,


    'pct_tl_nvr_dlq': 100.0, # indicating no history of delinquency
    
    # Used medians. They represent healthy credit seeking behavior.
    'open_acc_6m': 1, 
    'acc_open_past_24mths': 4.0,
    'open_acc': 11,
    'total_acc': 23,
    'revol_util': 53.9,
    'total_rev_hi_lim': 23600.0, 
    'open_act_il': 2.0,
    'mort_acc': 1.0,
    'num_rev_accts': 13.0,
    'mths_since_rcnt_il': 12.0,
    'mths_since_recent_bc': 12.0,
    'open_il_12m': 1.0,
    'open_il_24m': 1.0,
    'open_rv_12m': 1.0,
    'open_rv_24m': 2.0,
    'mths_since_recent_inq': 7.0,
    'num_actv_rev_tl': 5.0,
    'num_il_tl': 6.0,
    'total_bc_limit': 14000.0,


    # medians close to maxed out. Hence using industry sweet spot, a value between 30 - 40%
    'il_util': 35,
    'all_util': 30,

    # Using conservative baselines for age of credit history. 
    'credit_hist_age_mths': 48,
    'mo_sin_old_rev_tl_op': 60.0, 
    'mo_sin_old_rev_tl_op': 60.0, 
    'mo_sin_old_il_acct': 48.0,
    'rev_bal_to_inc': 0.05,

    # Current US market conditions
    'unemployment_rate': 4.4, 
    'fed_funds_rate': 3.64,  
    
    # is_thin_file set to 0. 
    # This indicated that "This borrower has enough historical data for the prediction to be high-confidence." 
    # A "Thin File" (1) usually triggers higher uncertainty and more conservative scoring.
    'is_thin_file': 0,

    # New applicants will always use modern metrics, so we can set this to 1 to reflect the current data environment.
    'has_modern_metrics': 1,
    
    'percent_bc_gt_75': 0.0 # applicant not maxing out card
    }

def calculate_derived_metrics(user_data):
    monthly_income = user_data['annual_inc'] / 12

    emp_len = user_data.get('emp_length', 'n/a')
    user_data['emp_unemployment_risk'] = get_employment_risk(emp_len)
    
    user_data['dti'] = (user_data.get('monthly_debt_payments', 0) / (monthly_income + 1)) * 100
    
    # Behavioral Flags (Derived from 'mths_since' features)
    # A 'recent' event is usually defined as within the last 12 months
    user_data['has_recent_bc'] = 1.0 if user_data.get('mths_since_recent_bc', 999) <= 12 else 0.0
    user_data['has_recent_inq'] = 1.0 if user_data.get('mths_since_recent_inq', 999) <= 6 else 0.0

    income = user_data.get('annual_inc', 50000)
    user_data['tot_hi_cred_lim'] =  income* 1.5
    user_data['avg_cur_bal'] = income * 0.15
    user_data['total_bal_ex_mort'] = income * 0.4
    user_data['tot_hi_cred_lim'] = income * 1.5
    user_data['total_bal_ex_mort'] = income * 0.4
    user_data['total_bal_il'] = income * 0.25
    user_data['max_bal_bc'] = income * 0.1
    
    return user_data

def get_employment_risk(emp_length):
    # Higher number = higher risk
    mapping = {
        '10+ years': 15.0,
        '9 years': 18.0,
        '5 years': 25.0,
        '1 year': 35.0,
        '< 1 year': 45.0,
        'n/a': 60.0 # Unemployed or not provided
    }
    return mapping.get(emp_length, 24.4) # Default to median 24.4

def get_column_names(preprocessor):
    num_features = preprocessor.transformers_[0][2].tolist()
    cat_transformer = preprocessor.named_transformers_['cat']
    cat_features_orig = preprocessor.transformers_[1][2]
    cat_features_transformed = cat_transformer.get_feature_names_out(cat_features_orig).tolist()
    return num_features + cat_features_transformed

def make_decision(probability):
    if probability < LOWER_THRESHOLD:
        return "AUTOMATIC ACCEPT", "Applicant qualifies for immediate approval."
    elif probability > UPPER_THRESHOLD:
        return "AUTOMATIC REJECT", "Risk exceeds acceptable banking parameters."
    else:
        return "MANUAL REVIEW", "Borderline risk; requires manual underwriter assessment."

def generate_human_explanation(model_results, client):
    prompt = f"""
    Acting as a Senior Credit Officer, provide a concise 2-sentence internal summary:
    - Decision: {model_results['decision']}
    - PD Score: {model_results['probability_of_default'] * 100:.2f}%
    - Key Risk Factors: {', '.join(model_results['top_drivers'])}
    
    Explain why this specific decision segment was chosen based on the data.
    """
    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return response.text.strip()
    except Exception as e:
        return f"Narrative unavailable: {str(e)}"

def predict_with_explanation(user_inputs):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found. Run train.py first.")
    
    pipeline = joblib.load(MODEL_PATH)
    classifier = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    full_data = user_inputs.copy()
    full_data = calculate_derived_metrics(full_data)
    
    for key, val in MOCK_BUREAU_DATA.items():
        if key not in full_data:
            full_data[key] = val
            
    df = pd.DataFrame([full_data])
    preprocessor = pipeline.named_steps['preprocessor']
    cat_cols = preprocessor.transformers_[1][2]

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    for col in pipeline.feature_names_in_:
        if col not in df.columns:
            df[col] = 0.0 if col not in cat_cols else "Unknown"
            
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
        impact = "Lowered Risk" if shap_val < 0 else "Raised Risk"
        detailed_reasons.append(f"{feat_name.split('__')[-1]}: ({impact})")

    results = {
        "probability_of_default": round(float(prob_default), 4),
        "decision": decision,
        "top_drivers": detailed_reasons
    }

    client = genai.Client(api_key=GEMINI_API_KEY)
    results["narrative"] = generate_human_explanation(results, client)

    return results

if __name__ == "__main__":
    sample_user_input = {
        'loan_amnt': 15000,
        'term': ' 36 months',
        'annual_inc': 75000,
        'home_ownership': 'MORTGAGE',
        'purpose': 'debt_consolidation',
        'addr_state': 'CA',
        'fico_range_low': 710,
        'monthly_debt_payments': 800, 
        'revol_bal': 12000,
        'emp_length': '5 years'
    }
    
    res = predict_with_explanation(sample_user_input)
    
    print("\n" + "="*60)
    print("           CREDIT DECISIONING ENGINE - ASSESSMENT")
    print("="*60)
    print(f"DECISION:        {res['decision']}")
    print(f"RISK SCORE (PD): {res['probability_of_default'] * 100:.2f}%")
    print("-" * 60)
    print("TOP PREDICTIVE FACTORS:")
    for reason in res['top_drivers']:
        print(f" • {reason}")
    print("-" * 60)
    print(f"OFFICER SUMMARY:\n{res['narrative']}")
    print("="*60)