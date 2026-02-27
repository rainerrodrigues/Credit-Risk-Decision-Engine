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
LOWER_THRESHOLD = 0.35  
UPPER_THRESHOLD = 0.75  
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

    # Realistic median defaults for Lending Club features (prevents model bias from zeros)
    NUMERIC_DEFAULTS = {
        'loan_amnt': 12000, 'int_rate': 12.0, 'annual_inc': 65000,
        'dti': 17.5, 'delinq_2yrs': 0, 'fico_range_low': 690, 'fico_range_high': 694,
        'inq_last_6mths': 0, 'mths_since_last_delinq': 36, 'mths_since_last_record': 80,
        'open_acc': 11, 'pub_rec': 0, 'revol_util': 50.0, 'total_acc': 25,
        'collections_12_mths_ex_med': 0, 'mths_since_last_major_derog': 60,
        'acc_now_delinq': 0, 'tot_coll_amt': 0, 'open_acc_6m': 1,
        'open_act_il': 2, 'open_il_12m': 1, 'open_il_24m': 2,
        'mths_since_rcnt_il': 8, 'total_bal_il': 18000, 'il_util': 65,
        'open_rv_12m': 2, 'open_rv_24m': 4, 'max_bal_bc': 5000,
        'all_util': 55, 'total_rev_hi_lim': 28000, 'inq_fi': 1,
        'total_cu_tl': 5, 'inq_last_12m': 2, 'acc_open_past_24mths': 4,
        'avg_cur_bal': 12000, 'chargeoff_within_12_mths': 0,
        'delinq_amnt': 0, 'mo_sin_old_il_acct': 120, 'mo_sin_old_rev_tl_op': 150,
        'mort_acc': 1, 'mths_since_recent_bc': 12, 'mths_since_recent_bc_dlq': 40,
        'mths_since_recent_inq': 4, 'num_accts_ever_120_pd': 0,
        'num_actv_rev_tl': 5, 'num_il_tl': 8, 'num_rev_accts': 14,
        'num_tl_120dpd_2m': 0, 'num_tl_30dpd': 0, 'num_tl_90g_dpd_24m': 0,
        'num_tl_op_past_12m': 2, 'pct_tl_nvr_dlq': 95.0,
        'percent_bc_gt_75': 20, 'pub_rec_bankruptcies': 0,
        'tax_liens': 0, 'tot_hi_cred_lim': 80000, 'total_bal_ex_mort': 35000,
        'total_bc_limit': 15000, 'unemployment_rate': 4.5, 'fed_funds_rate': 2.0,
        'revol_bal_to_inc': 0.3, 'emp_unemployment_risk': 0.05,
        'credit_hist_age_mths': 180,
    }

    for col in pipeline.feature_names_in_:
        if col not in df.columns:
            num_features = preprocessor.transformers_[0][2].tolist()
            if col in num_features:
                df[col] = NUMERIC_DEFAULTS.get(col, 0)
            else:
                df[col] = 'Unknown'

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