import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = {
    "StudyHours": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10],
    "Attendance": [50, 55, 60, 62, 65, 68, 70, 72, 75, 78, 80, 82, 85, 88, 92, 95],
    "Pass": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print("Student Dataset:")
print(df)

X = df[["StudyHours", "Attendance"]]
y = df["Pass"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

model = LogisticRegression(random_state=42)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("\n--- Logistic Regression Performance ---")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")

study_hours = float(input("\nEnter study hours: "))
attendance = float(input("Enter attendance percentage: "))

new_student = pd.DataFrame(
    {
    "StudyHours": [study_hours],
    "Attendance": [attendance]
    }
)

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("\nPrediction: PASS")
else:
    print("\nPrediction: FAIL")
