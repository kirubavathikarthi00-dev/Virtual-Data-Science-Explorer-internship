"""
Week 3 Task: Predictive Modeling and Algorithm Selection
Data Science with Python — Customer Churn Prediction

This script builds and compares three classification algorithms
(Logistic Regression, Decision Tree, Random Forest) to predict
customer churn, using the cleaned dataset from Week 2.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

sns.set_style("whitegrid")
RANDOM_STATE = 42

# -----------------------------
# 1. Load cleaned dataset (from Week 2)
# -----------------------------
df = pd.read_csv("customer_churn_cleaned.csv")
print("Dataset shape:", df.shape)

# Target variable
df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

# -----------------------------
# 2. Feature selection
# -----------------------------
# Based on Week 2 EDA: Contract and tenure were the strongest observed drivers.
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
categorical_features = ["Contract", "InternetService", "TechSupport", "PaymentMethod"]

X = df[numeric_features + categorical_features]
y = df["ChurnFlag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# -----------------------------
# 3. Preprocessing pipeline
# -----------------------------
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# -----------------------------
# 4. Define candidate models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE),
}

results = []
roc_data = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    # 5-fold cross-validation on training data
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")

    # Fit on full training set, evaluate on held-out test set
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        "Model": name,
        "CV Accuracy (mean)": round(cv_scores.mean(), 3),
        "Test Accuracy": round(acc, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1 Score": round(f1, 3),
        "ROC AUC": round(auc, 3),
    })

    roc_data[name] = roc_curve(y_test, y_proba)

results_df = pd.DataFrame(results)
print("\nModel comparison:")
print(results_df)
results_df.to_csv("model_comparison_results.csv", index=False)

# Select the best-performing model based on ROC AUC (most robust metric for
# imbalanced classification problems like churn prediction)
best_model_name = results_df.loc[results_df["ROC AUC"].idxmax(), "Model"]
print(f"\nBest model selected (by ROC AUC): {best_model_name}")

best_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", models[best_model_name])])
best_pipe.fit(X_train, y_train)
y_pred_best = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

# -----------------------------
# 5. Visualizations
# -----------------------------

# Figure 1: Model comparison bar chart (Accuracy, F1, AUC)
plt.figure(figsize=(8, 4.5))
x = np.arange(len(results_df))
width = 0.25
plt.bar(x - width, results_df["Test Accuracy"], width, label="Accuracy", color="#1565C0")
plt.bar(x, results_df["F1 Score"], width, label="F1 Score", color="#2E7D32")
plt.bar(x + width, results_df["ROC AUC"], width, label="ROC AUC", color="#EF6C00")
plt.xticks(x, results_df["Model"])
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Performance Comparison on Test Set")
plt.legend()
plt.tight_layout()
plt.savefig("fig4_model_comparison.png", dpi=150)
plt.close()

# Figure 2: Confusion matrix for best model (Random Forest)
plt.figure(figsize=(5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title(f"Confusion Matrix — {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("fig5_confusion_matrix.png", dpi=150)
plt.close()

# Figure 3: ROC curves for all models
plt.figure(figsize=(6.5, 5))
for name, (fpr, tpr, _) in roc_data.items():
    auc_val = results_df.loc[results_df["Model"] == name, "ROC AUC"].values[0]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("fig6_roc_curves.png", dpi=150)
plt.close()

# -----------------------------
# 6. Feature importance / coefficients (best model)
# -----------------------------
feature_names = (
    numeric_features +
    list(best_pipe.named_steps["preprocess"].named_transformers_["cat"].get_feature_names_out(categorical_features))
)
fitted_model = best_pipe.named_steps["model"]
if hasattr(fitted_model, "feature_importances_"):
    importances = fitted_model.feature_importances_
    imp_label = "Importance"
else:
    # Logistic Regression: use absolute coefficient magnitude as importance proxy
    importances = np.abs(fitted_model.coef_[0])
    imp_label = "Absolute Coefficient (Standardised)"

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)

plt.figure(figsize=(7, 4.5))
feat_imp.sort_values().plot(kind="barh", color="#2E7D32")
plt.title(f"Top 10 Feature Importances — {best_model_name}")
plt.xlabel(imp_label)
plt.tight_layout()
plt.savefig("fig7_feature_importance.png", dpi=150)
plt.close()

print("\nTop features:")
print(feat_imp)

# -----------------------------
# 7. Predict on hypothetical scenarios
# -----------------------------
hypothetical = pd.DataFrame([
    {"tenure": 2, "MonthlyCharges": 95, "TotalCharges": 190, "SeniorCitizen": 0,
     "Contract": "Month-to-month", "InternetService": "Fiber optic",
     "TechSupport": "No", "PaymentMethod": "Electronic check"},
    {"tenure": 48, "MonthlyCharges": 60, "TotalCharges": 2880, "SeniorCitizen": 0,
     "Contract": "Two year", "InternetService": "DSL",
     "TechSupport": "Yes", "PaymentMethod": "Credit card"},
    {"tenure": 12, "MonthlyCharges": 75, "TotalCharges": 900, "SeniorCitizen": 1,
     "Contract": "One year", "InternetService": "Fiber optic",
     "TechSupport": "No", "PaymentMethod": "Mailed check"},
])
hyp_proba = best_pipe.predict_proba(hypothetical)[:, 1]
hypothetical["Predicted_Churn_Probability"] = hyp_proba.round(3)
print("\nHypothetical scenario predictions:")
print(hypothetical[["tenure", "Contract", "MonthlyCharges", "Predicted_Churn_Probability"]])
hypothetical.to_csv("hypothetical_predictions.csv", index=False)

print("\nAll models trained, evaluated, and figures saved successfully.")
