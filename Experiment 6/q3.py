
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

depths = [1, 2, 3, 4, 5, 10, None]
train_acc = []
test_acc = []

for depth in depths:
    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=depth,
        random_state=42
    )

    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    train_acc.append(accuracy_score(y_train, y_train_pred))

    y_test_pred = model.predict(x_test)
    test_acc.append(accuracy_score(y_test, y_test_pred))

for d, tr, te in zip(depths, train_acc, test_acc):
    print(f"Depth: {str(d):>4} | "
          f"Training Accuracy: {tr:.4f} | "
          f"Testing Accuracy: {te:.4f}")

depth_labels = [str(d) for d in depths]

plt.figure(figsize=(10, 6))
plt.plot(depth_labels, train_acc, marker='o',
         label='Training Accuracy')
plt.plot(depth_labels, test_acc, marker='s',
         label='Testing Accuracy')

plt.xlabel("Maximum Tree Depth")
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy (Entropy)")
plt.ylim(0.8, 1.05)
plt.grid(True)
plt.legend()
plt.show()
