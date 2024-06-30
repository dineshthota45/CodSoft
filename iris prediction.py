# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Specify the path to your Iris dataset here
path_to_dataset = "C:/Users/dines/OneDrive/Desktop/iris.csv"  # Replace with your actual path

# Load the Iris dataset from CSV
df = pd.read_csv(path_to_dataset)

# Display the first few rows of the dataset
print(df.head())

# Separate features (X) and target (y)
X = df.drop('species', axis=1)  # Assuming 'species' column is the target
y = df['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data visualization - Histograms of features
plt.figure(figsize=(10, 6))
for i, feature in enumerate(X.columns):
    plt.subplot(2, 2, i + 1)
    plt.hist(X[feature], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel(feature)
    plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Decision Tree Model
# Initialize the Decision Tree classifier
dt_model = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree model
dt_model.fit(X_train, y_train)

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(dt_model, feature_names=X.columns, class_names=df['species'].unique(), filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# Make predictions with Decision Tree
y_pred_dt = dt_model.predict(X_test)

# Evaluate Decision Tree model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
cm_dt = confusion_matrix(y_test, y_pred_dt)
report_dt = classification_report(y_test, y_pred_dt, target_names=df['species'].unique())

print("Decision Tree Model Metrics:")
print(f"Accuracy: {accuracy_dt:.2f}")
print(f"Confusion Matrix:\n{cm_dt}")
print(f"Classification Report:\n{report_dt}")

# Random Forest Model
# Initialize the Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest model
rf_model.fit(X_train, y_train)

# Visualize feature importance
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Make predictions with Random Forest
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf, target_names=df['species'].unique())

print("\nRandom Forest Model Metrics:")
print(f"Accuracy: {accuracy_rf:.2f}")
print(f"Confusion Matrix:\n{cm_rf}")
print(f"Classification Report:\n{report_rf}")
