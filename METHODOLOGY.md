# Methodology: Churn Prediction Model

A plain-English walkthrough of how the logistic regression churn model was built, evaluated, and connected to the Power BI dashboard. Written to be understandable without a data science background, while staying technically accurate.

---

## 1. Data Cleaning

Every blank or missing value needs to be investigated before deciding how to handle it — a blank isn't automatically "missing data." In this dataset, 11 customers had a blank `TotalCharges` value; all 11 had `tenure = 0` (brand new customers who hadn't been billed yet), so the blanks were filled with `0`, not an average or a guess.

Identifier columns (like `customerID`) were removed before modeling, since they're labels, not predictive signals — but the ID was kept aside separately, so predictions could still be traced back to individual customers later.

## 2. Encoding Categorical Data

Machine learning models only understand numbers, not text categories like "Month-to-month" or "Fiber optic." Each categorical column was converted into a set of binary (0/1) columns — one per category — using one-hot encoding.

For a column with N categories, only N-1 binary columns are needed. For example, `Contract` (Month-to-month / One year / Two year) became just two columns: `Contract_One year` and `Contract_Two year`. If a customer has 0 in both, they're on a Month-to-month contract by elimination — no third column needed, and no artificial ranking implied between categories.

## 3. Train/Test Split

To evaluate a model honestly, it must be tested on data it never saw during training — otherwise it's an open-book exam. The data was split 80/20: 5,634 customers to train on, 1,409 held back purely for testing.

The split was stratified (`stratify=y`), meaning both the training and test sets preserve the same 26.5% churn rate found in the full dataset — this avoids the risk of an unlucky split skewing evaluation results.

## 4. Feature Scaling

Columns like `tenure` (0–72) and `MonthlyCharges` (18–120) sit on very different numeric scales. Without adjustment, a model can mistake "bigger raw numbers" for "more important," purely due to units. `StandardScaler` was used to convert every numeric column into a z-score — how many standard deviations a value sits from that column's average — putting every feature on a comparable scale before training.

## 5. Model Training — Logistic Regression

Logistic regression assigns a weight to every feature, learned automatically from historical data on who churned and who didn't. Positive weights push a prediction toward "churn," negative weights push toward "stays." The model is trained (`.fit()`) exclusively on the 80% training set.

## 6. Evaluation

**Accuracy alone is misleading here.** Since only 26.5% of customers churn, a model that predicts "no churn" for everyone would already be "73.5% accurate" while being completely useless. The trained model reached **80.7% accuracy** — a modest-looking improvement that hides the real story, which is why several additional metrics were used:

| Metric | Score | What it means |
|---|---|---|
| Precision | 65.8% | Of customers flagged as high-risk, 65.8% actually churned |
| Recall | 56.7% | Of customers who actually churned, 56.7% were successfully flagged |
| ROC-AUC | 0.842 | A random churner is ranked riskier than a random non-churner 84.2% of the time |

**ROC-AUC is the headline metric** for this project, since it evaluates the model's ranking ability across every possible decision threshold, rather than depending on one arbitrary cutoff (by default, a customer is flagged "churn" if their predicted probability is ≥ 50%).

Precision and recall trade off against each other — flagging more people as at-risk catches more real churners (higher recall) but also creates more false alarms (lower precision). Which balance is "right" is a business decision (cost of a missed customer vs. cost of a wasted retention call), not a purely mathematical one.

## 7. Feature Importance

Ranking every feature's weight reveals what the model leans on most:

**Strongest churn-risk drivers:** Fiber optic internet service, higher total charges, streaming service add-ons
**Strongest retention factors:** longer tenure (by far the single strongest factor), higher monthly charges, two-year contracts

**A note on interpretability:** `tenure`, `MonthlyCharges`, and `TotalCharges` are correlated with each other (customers who stay longer naturally accumulate higher total charges), which can make an individual feature's weight look counter-intuitive in isolation — a known statistical effect called multicollinearity. This doesn't affect the model's overall predictive accuracy, but it does mean individual coefficients should be interpreted with some caution rather than read too literally in isolation.

## 8. Exporting Predictions for Power BI

Each of the 1,409 test customers was scored with a churn probability (0–100%) and grouped into a simple risk tier:

- **High Risk** (≥70%): 89 customers
- **Medium Risk** (40–69%): 352 customers
- **Low Risk** (<40%): 968 customers

This output (`customer_churn_predictions.csv`) was imported into Power BI and related to the existing customer table via `CustomerID`, powering a dedicated "Predictive Risk Scoring" dashboard page — turning the model's output into something a retention team could act on directly, without needing to open Python.

---

## Tools Used
Python (pandas, scikit-learn) for data processing and modeling · Power BI for visualization and business-facing delivery

## Limitations & Honest Caveats
- Predictions cover the 1,409-customer test set only, not the full customer base
- Multicollinearity affects the literal interpretation of a few individual feature weights (see Section 7)
- The 50% classification threshold used for Accuracy/Precision/Recall is a default choice, not a business-optimized one — it could be adjusted based on the real-world cost of a missed customer vs. a false alarm
