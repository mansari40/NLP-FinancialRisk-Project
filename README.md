# Predicting Financial Risk Using NLP: A Sentiment- and Entity-Based Approach

This project explores the extent to which **natural language in financial news** can predict **company-level risk**. By combining **sentiment analysis**, **named entity recognition (NER)**, and **topic modeling**, we create **risk scores** from news headlines and validate them against real-world stock market behavior, especially during periods of volatility.

We then use this enriched feature set to train **machine learning classifiers** that predict short-term financial risk, rigorously evaluating model performance via **confusion matrices** and **ROC-AUC curves**.

The broader goal is to create a repeatable and scalable NLP framework that empowers both institutional and retail investors to gauge emerging risks in real time, driven by language signals in financial media.

## Research Question
To what extent can financial risk at the company level be anticipated and measured using sentiment analysis and named entity recognition in financial news, and how closely do these sentiment-based risk scores match the behavior of stock markets over time?

## Data Sources

News Headlines (Web-Scraped): Google News and Yahoo Finance headlines on selected companies (Tesla, Apple, Nvidia, Amazon) from the past 90 days.

Stock Prices: Historical daily price data from Yahoo Finance using yfinance, used to validate the sentiment-based risk signals with actual market behavior.

## Tools and Technologies

Python

## Natural Language Processing Libraries

- **Language Models**: `FinBERT`, `VADER`
- **NER**: `spaCy`
- **Transformers** `for FinBERT sentiment classification`
- **Topic Modeling**: `gensim`, `LDA`
- **Visualization**: `matplotlib`, `seaborn`
- **ML & Evaluation**: `scikit-learn`, `XGBoost`, `SMOTE`, `GridSearchCV`
- **Other Libraries**: `pandas`, `nltk`, `tqdm`, `yfinance`


## Methodology

## Data Collection

- Scraped financial news articles using **custom keyword queries**.
- Preprocessed articles to remove duplicates, normalize text, and extract clean headlines.

## Sentiment Analysis

- Used **FinBERT** for domain-specific sentiment classification.
- Compared against **VADER** for rule-based evaluation.
- Agreement/disagreement analysis and correlation with market metrics.
- Computed daily sentiment polarity features.

## Named Entity Recognition

- Applied `spaCy` to extract **entity types** (e.g., ORG, GPE, DATE, MONEY).
- Analyzed how different entities correlate with risk-laden headlines.

## Topic Modeling

- Performed **LDA topic modeling** per company.
- Generated daily topic distributions (as numeric features).
- Merged topics with financial data for each company-date pair.

## Risk Engineering

- Engineered a custom **risk score**:
  - Heavily weighted toward **negative sentiment** and **volatility**.
  - Offset slightly by **positive sentiment**.
- Labeled **“high-risk” days** based on **sharp negative price changes**.

---

## Predictive Modeling

### Models Used:
- **Logistic Regression**
- **Random Forest**
- **XGBoost**

### Key Steps:
- Features: Combined sentiment, topic, and volatility features.
- Target: Binary risk label (`1 = high risk`, `0 = normal`).
- Class Imbalance: Addressed with **SMOTE oversampling**.
- Evaluation:
  - Confusion Matrices
  - Precision, Recall, F1-Score
  - **ROC-AUC Comparison**
- Hyperparameter Tuning:
  - Used **GridSearchCV** with **StratifiedKFold** to tune all models.
  - Selected best parameters for robust performance on unseen data.

---

## 📈 Results

We evaluated all models on a hold-out test set after tuning them using **GridSearchCV** on a SMOTE-balanced dataset. Below are the final performance metrics and the best hyperparameters for each model:

### Final Model Performance (Test Set)

| Model                   | Accuracy | Precision | Recall | F1 Score | AUC  |
|-------------------------|----------|-----------|--------|----------|------|
| Logistic Regression (Tuned) | 0.72     | 0.71      | 0.74   | 0.73     | 0.73 |
| Random Forest (Tuned)       | 0.69     | 0.63      | 0.91   | 0.74     | 0.76 |
| XGBoost (Tuned)             | **0.78** | **0.73**  | **0.89** | **0.80** | **0.83** |

> **XGBoost consistently outperformed the other models across all evaluation metrics.**

---

### Best Hyperparameters per Model

| Model                | Best Parameters |
|----------------------|-----------------|
| Logistic Regression  | `C=0.01`, `penalty='l2'`, `solver='liblinear'` |
| Random Forest        | `n_estimators=100`, `max_depth=6`, `min_samples_split=10`, `min_samples_leaf=2`, `class_weight='balanced'` |
| XGBoost              | `n_estimators=100`, `max_depth=6`, `learning_rate=0.05`, `subsample=1.0`, `colsample_bytree=1.0`, `gamma=0`, `reg_lambda=1`, `scale_pos_weight=1.0` |

## **⚙ Requirements**

Install required libraries with:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost gensim tensorflow
