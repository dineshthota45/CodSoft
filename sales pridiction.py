#codsoft
#TASK-4 SALES PREDICTION USING PYTHON
#done by dinesh thota



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data from a specific directory
file_path = 'C:/Users/dines/OneDrive/Desktop/advertising.csv'
data = pd.read_csv(file_path)

# Verify column names and their presence in the DataFrame
expected_columns = ['TV', 'Radio', 'Newspaper', 'Sales']
for col in expected_columns:
    if col not in data.columns:
        raise KeyError(f"Column '{col}' is missing in the DataFrame.")

# Display histograms for each column
plt.figure(figsize=(12, 10))

# Histogram for 'TV'
plt.subplot(2, 2, 1)
plt.hist(data['TV'], bins=20, color='skyblue', edgecolor='black')
plt.title('TV Advertisement Spending')
plt.xlabel('TV Spending')
plt.ylabel('Frequency')

# Histogram for 'Radio'
plt.subplot(2, 2, 2)
plt.hist(data['Radio'], bins=20, color='salmon', edgecolor='black')
plt.title('Radio Advertisement Spending')
plt.xlabel('Radio Spending')
plt.ylabel('Frequency')

# Histogram for 'Newspaper'
plt.subplot(2, 2, 3)
plt.hist(data['Newspaper'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Newspaper Advertisement Spending')
plt.xlabel('Newspaper Spending')
plt.ylabel('Frequency')

# Histogram for 'Sales'
plt.subplot(2, 2, 4)
plt.hist(data['Sales'], bins=20, color='gold', edgecolor='black')
plt.title('Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')

# Adjust layout
plt.tight_layout()

# Show all plots
plt.show()

# Split data into features (X) and target (y)
X = data[['TV', 'Radio', 'Newspaper']]  # Features
y = data['Sales']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Visualizations
# Scatter plot of Predicted vs Actual Sales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2, label='Ideal Prediction')
plt.title('Predicted vs Actual Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.legend()
plt.grid(True)
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color='green')
plt.title('Residual Plot')
plt.xlabel('Predicted Sales')
plt.ylabel('Residuals')
plt.axhline(y=0, color='red', linestyle='--', lw=2)
plt.grid(True)
plt.show()
