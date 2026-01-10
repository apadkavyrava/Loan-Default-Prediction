## Objective

The goal of this project is to predict the probability of loan default for new applicants in the UK using historical credit and financial data. The project covers the full machine learning workflow, including:

- Exploratory data analysis (EDA)  
- Feature analysis and selection  
- Geospatial analysis  
- Model selection and optimization  
- Model evaluation using both classification and ranking metrics  

The main business objective is not only to classify defaults, but to **rank applicants by credit risk**, which is more realistic for lending decisions.

---

## Data

The original data consisted of two datasets:

1. Applicant financial and credit features with loan status  
2. Feature description metadata  

The dataset quality was generally high:

- No significant missing values  
- No major invalid entries  
- Dataset size: 237,436 observations, sufficient for model training and validation 

## Exploratory Data Analysis

### Target Distribution

Loan status is highly imbalanced:

- Defaults and charge-offs are rare events  
- Charge-off rate is approximately 6.3%  

This makes default prediction inherently difficult and requires special handling of class imbalance.

<img width="1156" height="336" alt="Screenshot 2026-01-10 at 17 26 56" src="https://github.com/user-attachments/assets/69653564-e3f0-4edb-a0e4-ee452baaf93e" />

From a business perspective:

- Charge-off is worse than default, as it reflects realized financial loss  
- Late >90 days is often a precursor to default but can also reflect restructuring or administrative delays  

For modeling purposes, these outcomes were later grouped together as “bad” loans.

### Feature Distributions

Feature-wise default distributions show realistic and meaningful patterns.

<img width="698" height="667" alt="Screenshot 2026-01-10 at 17 28 38" src="https://github.com/user-attachments/assets/01bd71b5-f2f4-42b5-ac74-34ce7b1dd750" />


#### Strong Predictors

 - Credit behavior & delinquency:
 - Utilization & leverage:

These are classic and well-known predictors of default in credit risk modeling, indicating good data quality.

#### Weak or Noisy Predictors

- Employment length  
- Job title (after NLP parsing)  

These features showed high noise and weak separation, so they were excluded from final models.

### Correlation and Data Leakage Risks

<img width="1258" height="460" alt="Screenshot 2026-01-10 at 17 29 37" src="https://github.com/user-attachments/assets/e8c2681d-9438-4d2e-b95d-ab25266e6fa7" />

High correlations were observed with:

- Year of borrowing  
- Interest rate  

These variables are strongly influenced by macroeconomic conditions and lending policy changes over time. Including them can introduce **temporal data leakage**, allowing the model to learn time effects instead of borrower risk.

Heatmaps also suggested possible leakage from:

- `loan_amount`  
- `amount_paid`  
- `installment`  

These features describe the loan contract rather than applicant risk and may not be available at application time. They were therefore excluded in later modeling stages.

## PCA Analysis

<img width="607" height="538" alt="Screenshot 2026-01-10 at 17 30 53" src="https://github.com/user-attachments/assets/e947c696-c546-4c98-a5ce-03eeb61f1a1d" />

A 3D PCA visualization showed:

- No clear separation between default and non-default cases  
- Heavy overlap across loan statuses  

This indicates that:

- Default is not driven by a small number of linear directions  
- Risk is distributed across many interacting features  

This supports the use of **nonlinear models** such as tree-based methods rather than linear classifiers.

## Geospatial Analysis

Two types of maps were analyzed.

### Raw Loan Locations

[Interactive loan map](https://apadkavyrava.github.io/Loan-Default-Prediction/uk_loan_map.html)



- No clear geographic clusters of default  
- Most loans originate from large urban areas  

### County-Level Normalized Default Rates

<img width="690" height="1090" alt="image" src="https://github.com/user-attachments/assets/5e9228b9-70c6-401f-a860-e7f5d9c39b4c" />

- Some regional variation exists, mostly in rural areas  
- However, geographic features were weak compared to financial behavior variables  

Geo features were not included in final models unless required for incremental performance improvement.


# Data Preprocessing

### Target Engineering

Loan statuses were re-labeled into two classes:

- **0 = Non-default**  
- **1 = Default-like outcome (Default, Late >90 days, Charge-off)**  

Ongoing loans were excluded to avoid label uncertainty.

---

### Feature Processing

- All categorical variables were encoded  
- Job titles were parsed using spaCy but produced noisy results and were removed  
- Identifiers such as `account_id` were excluded  


## Model Selection and Training

Models tested:

- Random Forest  
- XGBoost  
- LightGBM  

### Training Pipeline

<img width="1112" height="112" alt="Screenshot 2026-01-10 at 17 47 08" src="https://github.com/user-attachments/assets/45b2d432-dee9-4409-bd6c-00ddabdb93e2" />


## Initial Results and Leakage Detection

The first XGBoost model produced:

- **Train AUC ≈ 0.99**  
- Near-perfect test set metrics  
- Stable cross-validation  

Although this initially looked excellent, feature importance revealed dominance of:

- `amount_paid`  
- `installment`  

These are strong indicators of **post-loan behavior**, confirming data leakage.

After removing all loan-structure and post-origination variables, the model was retrained using only applicant information available at decision time.


## Final Model Performance

With leakage removed:

- ROC AUC ≈ **0.69**  
- Minority class (default) recall and F1 were low due to strong class imbalance  
<img width="487" height="218" alt="Screenshot 2026-01-10 at 17 48 08" src="https://github.com/user-attachments/assets/0b660ecb-13ad-43ca-8462-2263b1a925f0" />

Using class-weighted training (no resampling) slightly improved minority class detection:

- Default-class F1 increased to ~**13%**

This level of performance is realistic for credit risk models using only application data.


## From Classification to Risk Ranking

Binary classification metrics alone are not sufficient for credit decisions.  
In lending, the goal is to **rank applicants by risk** and apply policies to high-risk segments.

Therefore, predicted probabilities were used to create risk buckets.

### Risk Ranking Evaluation

Applicants were sorted by predicted default probability:

| Risk Segment | Default Rate |
|-------------|-------------:|
| Top 1% (highest risk) | 24.8% |
| Top 2% | 23.2% |
| Top 5% | 19.7% |
| Top 10% | 16.9% |

Population default rate ≈ **6%**

This shows that:

- High-risk groups have approximately **3–4× higher default rates** than average  
- The model provides meaningful ranking signal, even if binary classification metrics are modest  


