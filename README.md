# Credit Risk Classification with SHAP Interpretability

**Binary Classification | LightGBM / XGBoost | Imbalanced Data Handling | Model Explainability**

Predicting loan defaults using simulated credit bureau data to support transparent and compliant lending decisions.

- **Dataset**: [Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Target**: `loan_status` (1 = default/charged off, \~22% positive class)
- **Best model**: LightGBM with class weighting and hyperparameter tuning → ROC-AUC ≈ 0.9481 on test set
- **Focus**: 
  - Effective handling of class imbalance  
  - SHAP-based interpretability to explain "why a loan was denied"

### Handling Imbalance
The target variable `loan_status` is imbalanced (\~21.8% defaults).  
Several strategies were applied:
- **Class weighting**: `class_weight='balanced'` in Logistic Regression and RandomForest; `scale_pos_weight` ≈ 3.59 in LightGBM.
- **Oversampling**: SMOTE applied to training data for separate RandomForest and LightGBM experiments.

### SHAP Explanations for "Why Denied"
SHAP (SHapley Additive exPlanations) provides transparent, feature-level insights crucial for financial decision-making and regulatory compliance.  

**Key Insights from SHAP**:
- `loan_int_rate`: Most influential — higher rates strongly increase default probability.
- `loan_percent_income`: High loan-to-income ratios significantly raise risk.
- `loan_grade`: Poor grades (D–G) are strong default predictors.
- `cb_person_default_on_file`: Prior default is a major risk driver.

**Business Recommendation**:  
Integrate SHAP values into the loan approval workflow. For denied or high-risk applications, generate clear explanations (e.g., "Application flagged due to high proposed interest rate and elevated loan-to-income ratio."). This promotes transparency for applicants, supports credit officer reviews, and aligns with regulatory requirements for explainable AI in lending.

## Key Results
- **Top SHAP features**: loan_int_rate, loan_percent_income, loan_grade, cb_person_default_on_file
- **Business value**: Transparent, auditable decisions for regulatory compliance and risk management

## Notebook
[credit_risk_xgboost_shap.ipynb](credit_risk_xgboost_shap.ipynb)

## How to Run
1. Download `credit_risk_dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
2. Open `credit_risk_xgboost_shap.ipynb` in Jupyter Notebook or Google Colab
3. Install dependencies (if needed):  
   `pip install pandas numpy scikit-learn imbalanced-learn lightgbm xgboost shap matplotlib seaborn`
4. Run all cells

## Visuals
![SHAP Summary Plot](images/shap_summary.png)  
![SHAP Force Plot Example](images/shap_force_example.png)
