import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

student_info = {
    "Hours": [2, 3, 1, 4, 5, 6, 3, 7, 8, 5, 9, 6, 10, 7, 4, 8],
    "Attendance_Percent": [52, 58, 55, 65, 70, 74, 63, 80, 85, 76, 94, 82, 96, 88, 68, 90],
    "Result": [0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1]
}

students = pd.DataFrame(student_info)

print("Student Information:")
print(students)

inputs = students[["Hours", "Attendance_Percent"]]
result = students["Result"]

train_inputs, test_inputs, train_result, test_result = train_test_split
(
    inputs,
    result,
    test_size=0.25,
    random_state=15,
    stratify=result
)

classifier = LogisticRegression()
classifier.fit(train_inputs, train_result)

test_prediction = classifier.predict(test_inputs)

acc = accuracy_score(test_result, test_prediction)
pre = precision_score(test_result, test_prediction)
rec = recall_score(test_result, test_prediction)
f1 = f1_score(test_result, test_prediction)

print("\n--- Classification Results ---")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {pre:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")

hours_input = float(input("\nEnter number of study hours: "))
attendance_input = float(input("Enter attendance percentage: "))

student = pd.DataFrame(
{
    "Hours": [hours_input],
    "Attendance_Percent": [attendance_input]
}
)

result_prediction = classifier.predict(student)

status = "PASS" if result_prediction[0] == 1 else "FAIL"

print("\nPrediction:", status)
