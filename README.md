# 🛡️ Credit Risk Decision Engine

An AI-powered credit risk assessment engine that leverages Machine Learning
(LightGBM) and Explainable AI (SHAP) to evaluate loan applications instantly. It
also uses Google's Gemini LLM to generate professional, data-driven narrative
summaries explaining the decision.

## ✨ Features

- **Instant Credit Risk Assessment:** Calculates the Probability of Default (PD)
  score based on borrower application data.
- **Automated Decisioning:** Automatically segregates applicants into "AUTOMATIC
  ACCEPT", "MANUAL REVIEW", or "AUTOMATIC REJECT" based on configured risk
  thresholds.
- **Explainable AI (XAI):** Uses SHAP (SHapley Additive exPlanations) to provide
  the top 3 critical risk drivers for every decision, ensuring transparency.
- **LLM-Powered Narratives:** Integrates with `gemini-2.5-flash` to generate a
  concise, human-readable executive summary of the credit decision.
- **Modern Interactive UI:** A sleek frontend dashboard with dark/light mode
  functionality, an interactive gauge chart, and responsive status indicators.

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Machine Learning:** Scikit-Learn, LightGBM, SHAP, Pandas, NumPy
- **LLM Integration:** Google GenAI (`gemini-2.5-flash`)
- **Frontend:** HTML5, CSS3 (Vanilla), JavaScript
- **Deployment:** Docker

## ⚙️ Prerequisites

- Python 3.11+ (if running locally)
- Docker (if running via containers)
- A Google Gemini API Key

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/rainerrodrigues/Credit-Risk-Decision-Engine.git
cd Credit-Risk-Decision-Engine
```

### 2. Set up environment variables

Create a `.env` file in the root directory and add your Gemini API key:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 3. Run Locally (Without Docker)

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI development server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

_Note: Ensure that the trained model (`optimized_credit_risk_model.joblib`) is
placed in the `Models/` directory relative to the project root before running._

### 4. Run With Docker

Build the Docker image:

```bash
docker build -t credit-risk-engine .
```

Run the container:

```bash
docker run -p 8000:8000 --env-file .env credit-risk-engine
```

## 🎯 Usage

1. Open your browser and navigate to `http://localhost:8000`.
2. Fill out the loan application form with the borrower's details.
3. Click "Evaluate Application" or "Run Sample Data".
4. View the interactive Credit Risk Assessment Report, including the PD score,
   decision, risk drivers, and the LLM-generated executive narrative.

## 📝 License

This project is licensed under the MIT License.
