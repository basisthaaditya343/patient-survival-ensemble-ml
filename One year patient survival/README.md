# 🧠 One-Year Patient Survival Prediction using Ensemble ML

This project develops a stacked-ensemble model to predict one-year patient survival using clinical data.

## 🚀 Highlights
- **Dataset:** ~25,000 anonymized records
- **Model:** 8 base learners + Logistic Regression meta-model
- **Performance:** Accuracy 84.07%, Precision 0.8655, Recall 0.8865, F1-score 0.8759
- **Explainable AI:** SHAP & LIME for model interpretability
- **Interface:** Gradio app for instant clinician-friendly predictions

## ⚙️ How to Run
```bash
git clone https://github.com/basisthaaditya343/one-year-patient-survival.git
cd one-year-patient-survival
pip install -r requirements.txt
python src/app.py