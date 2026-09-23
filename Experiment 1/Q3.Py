import pandas as pd

data = {
    "Student Name": ["Rahul", "Pritam", "Samrat", "Anubhav", "Ankit"],
    "Roll Number": [34, 33, 43, 11, 7],
    "Marks": [72, 85, 91, 77, 68],
    "Attendance": [88, 92, 76, 95, 81]
}

df = pd.DataFrame(data)

def assign_grade(marks):
    if marks >= 90:
        return 'A'
    elif marks >= 80:
        return 'B'
    elif marks >= 70:
        return 'C'
    else:
        return 'D'

df["Grade"] = df["Marks"].apply(assign_grade)
print(df)