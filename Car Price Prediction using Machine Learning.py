import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
file_path = '/mnt/data/car data.csv'
data = pd.read_csv(r"c:/Users/janan/OneDrive/Desktop/Oasis Infobyte/car data.csv")

# Drop 'Car_Name' as it doesn't add predictive value
data = data.drop(['Car_Name'], axis=1)

# Encode categorical columns
le = LabelEncoder()
for col in ['Fuel_Type', 'Selling_type', 'Transmission']:
    data[col] = le.fit_transform(data[col])

# Features (X) and Target (y)
X = data.drop(['Selling_Price'], axis=1)
y = data['Selling_Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Example: Predicting a new car price
example = np.array([[2018, 10.0, 15000, 1, 0, 1, 0]])  # Year, Present_Price, Driven_kms, Fuel_Type, Selling_type, Transmission, Owner
print("Predicted Selling Price:", model.predict(example)[0])
