import pandas as pd

student_data = {
    "Student Name": ["Arjun", "Priya", "Kunal", "Sneha", "Vikram"],
    "Roll Number": [12, 18, 23, 27, 34],
    "Marks": [76, 89, 73, 94, 82],
    "Attendance": [84, 91, 79, 96, 87]
}

student_df = pd.DataFrame(student_data)

selected_students = student_df[student_df["Marks"] > 80]

print("Students with Marks > 80:")
print(selected_students)
