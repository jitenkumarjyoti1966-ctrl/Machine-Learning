import pandas as pd

students = pd.DataFrame(
{
    "Student Name": ["Dev", "Ishita", "Varun", "Megha", "Nikhil"],
    "Roll Number": [26, 13, 31, 8, 45],
    "Marks": [83, 69, 92, 76, 88],
    "Attendance": [91, 84, 95, 79, 87]
}
)

def grade(score):
    grades = {
        90: "A",
        80: "B",
        70: "C",
        0: "D"
    }

    for minimum, letter in sorted(grades.items(), reverse=True):
        if score >= minimum:
            return letter

students["Grade"] = students["Marks"].map(grade)

print(students)
