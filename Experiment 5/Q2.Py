import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)

model.fit(X_train, y_train)

y_probability = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_probability)

auc_score = roc_auc_score(y_test, y_probability)

print("--- ROC and AUC ---")
print(f"AUC Score: {auc_score:.4f}")

plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, color="blue", linewidth=2, label=f"Logistic Regression (AUC = {auc_score:.4f})")

plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")

plt.legend()
plt.grid()

plt.show()
