import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

cancer_data = load_breast_cancer()

features = cancer_data.data
labels = cancer_data.target

train_features, test_features, train_labels, test_labels = train_test_split(
    features,
    labels,
    test_size=0.25,
    random_state=15,
    stratify=labels
)

normalizer = StandardScaler()

train_features = normalizer.fit_transform(train_features)
test_features = normalizer.transform(test_features)

classifier = LogisticRegression
(
    max_iter=1500,
    random_state=15
)

classifier.fit(train_features, train_labels)

probabilities = classifier.predict_proba(test_features)[:, 1]

false_positive, true_positive, cutoff = roc_curve
(
    test_labels,
    probabilities
)

roc_value = roc_auc_score
(
    test_labels,
    probabilities
)

print("--- ROC Curve Analysis ---")
print(f"Area Under Curve: {roc_value:.4f}")

plt.figure(figsize=(9, 6))

plt.plot
(
    false_positive,
    true_positive,
    color="green",
    linewidth=2,
    label=f"Logistic Model (AUC = {roc_value:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    color="black",
    linestyle=":",
    label="Random Prediction"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Breast Cancer Classification")
plt.legend()
plt.grid(alpha=0.3)

plt.show()
