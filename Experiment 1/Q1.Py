import numpy as np

marks = np.array([78, 65, 82, 90, 56, 74, 88, 69, 95, 71])
mean = np.mean(marks)
median = np.median(marks)
std = np.std(marks)
maximum = np.max(marks)
minimum = np.min(marks)

print("Internal Marks:", marks)
print("Mean:", mean)
print("Median:", median)
print("Standard Deviation:", std)
print("Maximum:", max)
print("Minimum:", min)

