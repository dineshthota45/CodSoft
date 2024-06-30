import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
file_path = 'C:/Users/dines/OneDrive/Desktop/Imdb_movie.csv'
df = pd.read_csv(file_path, encoding='latin1')

# Preview the dataset
print("First few rows of the dataset:")
print(df.head())

# Data Cleaning
# Remove rows with missing target values (Rating)
df = df.dropna(subset=['Rating'])

# Convert the 'Year' column to numeric, extract the numeric part if necessary
df['Year'] = df['Year'].str.extract('(\d{4})').astype(float)

# Convert 'Duration' to numeric (extract minutes)
df['Duration'] = df['Duration'].str.extract('(\d+)').astype(float)

# Convert 'Votes' to numeric (remove commas and convert to float)
df['Votes'] = df['Votes'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Feature selection: Use relevant features
features = ['Year', 'Duration', 'Genre', 'Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
X = df[features]
y = df['Rating']

# Data Visualization

# Distribution of Ratings
plt.figure(figsize=(10, 6))
sns.histplot(y, bins=30, kde=True)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Pairplot of numerical features
plt.figure(figsize=(10, 6))
sns.pairplot(df[['Year', 'Duration', 'Votes', 'Rating']])
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
corr_matrix = df[['Year', 'Duration', 'Votes', 'Rating']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Define preprocessing steps for numeric and categorical features
numeric_features = ['Year', 'Duration', 'Votes']
numeric_transformer = SimpleImputer(strategy='median')

categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(random_state=42))])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Display feature importance
model_named_steps = model.named_steps['regressor']
importances = model_named_steps.feature_importances_

# Get feature names after one-hot encoding
feature_names = numeric_features + list(model.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(categorical_features))

# Create a DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("Feature Importance:")
print(feature_importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()
