#codsoft
#Task-5 credit card fraud detection
#done by dinesh thota



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
file_path = 'C:/Users/dines/OneDrive/Desktop/creditcard.csv'
data = pd.read_csv(file_path)

# Display the first 10 rows of the dataset
print("First 10 rows of the dataset:")
print(data.head(10))

# Normalize the 'Amount' feature
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

# Plot histogram of 'Class' (0: Genuine, 1: Fraudulent)
plt.figure(figsize=(8, 6))
plt.hist(data['Class'], bins=2, color='skyblue', edgecolor='black', alpha=0.7)
plt.xticks([0, 1], ['Genuine', 'Fraudulent'])
plt.title('Distribution of Transactions')
plt.xlabel('Transaction Class')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# Handle class imbalance with oversampling
oversampler = RandomOverSampler(sampling_strategy='minority', random_state=42)
X = data.drop(['Time', 'Class'], axis=1)
y = data['Class']
X_over, y_over = oversampler.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train Logistic Regression Classifier
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

# Evaluate the models
models = [('Decision Tree', dt_model), ('Random Forest', rf_model), ('Logistic Regression', lr_model)]

for model_name, model in models:
    print(f"\n{model_name}:")
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=['Genuine', 'Fraudulent'], fontsize=10)
plt.title("Decision Tree Classifier")
plt.show()
