import numpy as np

marks = np.array([78, 65, 82, 90, 56, 74, 88, 69, 95, 71])

results = {
    "Mean": np.mean(marks),
    "Median": np.median(marks),
    "Standard Deviation": np.std(marks),
    "Maximum": np.max(marks),
    "Minimum": np.min(marks)
}

print("Internal Marks:", marks)

for label, value in results.items():
    print(label + ":", value)
