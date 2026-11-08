# Telco Customer Churn – Comparing Five Classifiers

## a. Problem statement
A telecom company loses recurring revenue every time a subscriber cancels, and retaining
an existing customer is far cheaper than acquiring a new one. The goal of this project is
to predict, from a customer's demographics, subscribed services and billing details,
whether that customer will churn (leave the company). Five classification models are
trained on the same dataset and compared on six evaluation metrics to identify which one
detects likely churners most reliably. The trained models are served through an
interactive Streamlit app where test data can be uploaded and evaluated.

## b. Dataset description
- **Source:** Telco Customer Churn (IBM sample dataset), Kaggle –
  https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Instances:** 7,043 customers
- **Features used:** 19 (after dropping the identifier `customerID`)

| Group | Features |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account and billing | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

- **Target:** `Churn` (Yes / No) – binary classification, "Yes" is the positive class
- **Class balance:** 26.5% Yes / 73.5% No. The data is imbalanced, so a model that
  always predicts "No" would already reach about 73.5% accuracy. MCC, F1 and recall are
  therefore more informative than accuracy for this problem.
- **Data quality issues handled:**
  - `TotalCharges` is stored as text and has 11 blank values (new customers with
    tenure 0). It is converted to numeric and the blanks are median-imputed.
  - `SeniorCitizen` is coded 0/1 and is treated as a Yes/No category like the other
    demographic columns.
- **Preprocessing:** standard scaling for the 3 numeric columns (`tenure`,
  `MonthlyCharges`, `TotalCharges`) and one-hot encoding for the 16 categorical columns.
  The data is split 80/20 with stratification on `Churn` (`random_state=42`), giving
  5,634 training and 1,409 test customers. The preprocessing is saved inside each model
  pipeline, so the app accepts raw rows in the original Kaggle format.

## c. GitHub repository
- **Repository:** https://github.com/gshivani0211/BITS_ML_Assignment_2_telco-churn-ml_2026AJ05787
- **Live app:** https://bitsmlassignment2telco-churn-ml2026aj05787-aohf4khptuv74fajmv6.streamlit.app

```
├── app.py               Streamlit app (loads saved models, no retraining)
├── train_models.py      Trains, evaluates and saves all five models
├── preprocessing.py     Telco-specific cleaning shared by training and the app
├── metrics_utils.py     Metric calculation shared by training and the app
├── requirements.txt
├── README.md
├── test_data.csv        Held-out 20% test split (1,409 customers)
├── data/                Raw Kaggle CSV
└── model/               Five saved .joblib pipelines, metadata.json, metrics.csv
```

### Streamlit app features
- Upload of a test CSV in the original Telco format
- Dropdown to select any of the five models
- Display of all six evaluation metrics for the selected model
- Confusion matrix and classification report
- Ranked table of the customers with the highest predicted churn probability
- Side-by-side comparison of all five models on the uploaded data

## d. Models used
| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=2000` |
| Decision Tree | `max_depth=6`, `min_samples_leaf=20` |
| K-Nearest Neighbors | `n_neighbors=15` |
| Naive Bayes | Gaussian |
| Support Vector Machine | RBF kernel, `probability=True` |

Precision, Recall and F1 are reported for the churn ("Yes") class.

### Comparison table (held-out test set, 1,409 customers – run on BITS Virtual Lab)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | **0.8055** | **0.8419** | 0.6572 | 0.5588 | **0.6040** | **0.4790** |
| Decision Tree | 0.7970 | 0.8383 | **0.6679** | 0.4679 | 0.5503 | 0.4357 |
| KNN | 0.7814 | 0.8223 | 0.5948 | 0.5535 | 0.5734 | 0.4272 |
| Naive Bayes | 0.6955 | 0.8075 | 0.4597 | **0.8396** | 0.5941 | 0.4268 |
| Support Vector Machine | 0.7906 | 0.7932 | 0.6386 | 0.4866 | 0.5524 | 0.4255 |

Bold marks the best value in each column.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best on four of six metrics: accuracy (0.8055), AUC (0.8419), F1 (0.604) and MCC (0.479). It catches 209 of the 374 churners and wrongly flags 109 loyal customers. Its strong result suggests the churn signal is largely linear: month-to-month contracts, short tenure and higher monthly charges push churn risk up in a fairly additive way. It is also fast and its coefficients are easy to interpret. |
| Decision Tree | Highest precision (0.6679): when it predicts churn it is usually right, with only about 87 false alarms. However, recall is low (0.4679), so it misses more than half of the actual churners. AUC (0.8383) is close to Logistic Regression, showing it ranks customers well, but the depth limit makes its yes/no cut-off conservative. |
| KNN | Recall close to Logistic Regression (0.5535, about 207 churners caught) but lower precision (0.5948), because it raises about 141 false alarms. MCC is lower (0.4272). With about 45 one-hot columns, distances between customers become less meaningful, which limits a distance-based method. |
| Naive Bayes | Lowest accuracy (0.6955), even below the 73.5% "always No" baseline, yet by far the highest recall (0.8396): it catches 314 of 374 churners. The cost is about 369 false alarms (precision 0.4597). Its independence assumption does not hold for this data (for example, the streaming and security columns are all tied to `InternetService`). Its F1 is still second best (0.5941), which shows clearly why accuracy alone is misleading on imbalanced data. |
| Support Vector Machine | Lowest AUC (0.7932) and MCC (0.4255), with fairly conservative predictions (recall 0.4866). It was run with default hyperparameters (C=1, RBF kernel); tuning C and gamma would likely improve it. It was also the slowest model to train. |
| **Overall winner** | **Logistic Regression.** It gives the best balance between catching churners and avoiding false alarms (highest F1 and MCC) and the best overall ranking ability (highest AUC), while being the simplest and most interpretable model. If the business goal were only to catch as many churners as possible and false alarms were cheap (for example, a low-cost retention email), Naive Bayes would be a reasonable alternative because of its 0.84 recall. |

## How to run
```bash
pip install -r requirements.txt
python train_models.py      # trains and saves the models, writes test_data.csv
streamlit run app.py        # opens the app at http://localhost:8501
```
Upload `test_data.csv` in the sidebar to evaluate the models.
