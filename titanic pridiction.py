# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and preprocess Titanic dataset for Random Forest
def load_and_preprocess_rf(path_to_dataset):
    # Load the dataset
    df = pd.read_csv(path_to_dataset)

    # Removing unnecessary columns
    df = df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns')

    # Convert gender to binary
    df['Gender'] = df['Sex'].map({'male': 0, 'female': 1})
    df = df.drop('Sex', axis='columns')

    # Handling missing values for column Age using fillna()
    df['Age'] = df['Age'].fillna(df['Age'].mean())

    return df

# Function to train and evaluate Random Forest model
def train_and_evaluate_rf(df):
    # Define target and input variables
    target = df['Survived']
    input_var = df.drop('Survived', axis='columns')

    # Split the dataset into training and testing parts
    X_train, X_test, y_train, y_test = train_test_split(input_var, target, test_size=0.2, random_state=42)

    # Initialize the Random Forest model
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the model with the training data
    model_rf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model_rf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix - Random Forest')
    plt.show()

    # Plot feature importance (assuming Random Forest)
    plt.figure(figsize=(10, 6))
    feat_importances = pd.Series(model_rf.feature_importances_, index=input_var.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title('Feature Importance - Random Forest')
    plt.show()

    # Plot histogram of Age distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.show()

    # Plot decision tree from Random Forest (first tree)
    plt.figure(figsize=(14, 10))
    plot_tree(model_rf.estimators_[0], filled=True, feature_names=input_var.columns, class_names=['Not Survived', 'Survived'])
    plt.title('Decision Tree - Random Forest (Sample Tree)')
    plt.show()

    return model_rf, accuracy, cm, report

# Paths to your datasets
path_to_rf_dataset = "C:/Users/dines/OneDrive/Desktop/Titanic-Dataset.csv"

# Load and preprocess dataset for Random Forest
df_rf = load_and_preprocess_rf(path_to_rf_dataset)

# Train and evaluate Random Forest model
model_rf, accuracy_rf, cm_rf, report_rf = train_and_evaluate_rf(df_rf)

# Display results for Random Forest
print("Random Forest Model Accuracy:", accuracy_rf)
print("Random Forest Confusion Matrix:\n", cm_rf)
print("Random Forest Classification Report:\n", report_rf)
