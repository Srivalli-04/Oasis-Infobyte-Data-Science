# Sales Prediction using Python (Regression)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# 1. Load the Dataset

df = pd.read_csv("advertising.csv")

print("Dataset Sample:")
print(df.head())

# -----------------------------
# 2. Data Exploration
# -----------------------------
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Check correlations
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# 3. Split Data
# -----------------------------
X = df[['TV', 'Radio', 'Newspaper']]   # Features
y = df['Sales']                        # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 4. Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# -----------------------------
# 6. Visualize Actual vs Predicted
# -----------------------------
plt.scatter(y_test, y_pred, color="blue")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# -----------------------------
# 7. Test with New Data
# -----------------------------
# Example: New ad spending [TV, Radio, Newspaper]
new_data = [[150, 20, 15]]
predicted_sales = model.predict(new_data)
print("\nPredicted Sales for [TV=150, Radio=20, Newspaper=15] : ", predicted_sales[0])
