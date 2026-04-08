import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
data = {
    "Income": [25000, 40000, 50000, 60000, 70000, 80000, 30000, 90000],
    "Credit_Score": [600, 650, 700, 720, 750, 800, 620, 780],
    "Age": [25, 35, 45, 32, 40, 50, 28, 48],
    "Approved": [0, 0, 1, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)
df
X = df[["Income", "Credit_Score", "Age"]]
y = df["Approved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("Model trained successfully!")
income = 55000
credit_score = 710
age = 30

prediction = model.predict([[income, credit_score, age]])

if prediction[0] == 1:
    print("Loan Approved ✅")
else:
    print("Loan Rejected ❌")
plt.figure(figsize=(10,6))
tree.plot_tree(model, feature_names=["Income", "Credit Score", "Age"], filled=True)
plt.show()
